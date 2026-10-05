"""Known event-dependent covariance -> profiled clock fit -> optical delay."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar
from optics import image, fit_center, SIGMA, PIXELS, READ_SD, ORIGIN, VELOCITY

RATE, OFFSET, DELAY, FRAME = 1.002, .030, .040, 2.0
SYNC = np.linspace(-1., 1., 40)
RATE_BOUNDS = (.5, 1.5)  # prior physical range, reference seconds/master second


def covariances(heterogeneity=1.):
    """Seconds squared; alternating high/low quality independent sync events."""
    if not np.isfinite(heterogeneity) or not 1 <= heterogeneity <= 8:
        raise ValueError('Require heterogeneity in [1,8]')
    h = np.where(np.arange(len(SYNC)) % 2, heterogeneity, 1.)
    sx, sy = .030*h, .020*np.sqrt(h)
    rho = .5-.1*np.log2(h)
    c = np.zeros((len(h),2,2))
    c[:,0,0], c[:,1,1] = sx*sx, sy*sy
    c[:,0,1] = c[:,1,0] = rho*sx*sy
    return c


def validate(x, y, c):
    x, y, c = np.asarray(x,float), np.asarray(y,float), np.asarray(c,float)
    if x.ndim != 1 or x.shape != y.shape or len(x)<3 or not np.all(np.isfinite([x,y])):
        raise ValueError('Require at least three finite timestamp pairs')
    if c.shape != (len(x),2,2) or not np.all(np.isfinite(c)) or not np.allclose(c,c.transpose(0,2,1),rtol=1e-12,atol=1e-15):
        raise ValueError('Require one finite symmetric 2x2 covariance per pair')
    eig = np.linalg.eigvalsh(c)
    if np.any(eig[:,0] <= 0) or np.any(eig[:,1]/eig[:,0] > 1e12):
        raise ValueError('Covariances must be well-conditioned positive definite')
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        raise ValueError('Timestamp span insufficient')
    return x,y,c


def profile(rate, x, y, c):
    """Eliminate offset at fixed rate; also accepts an array of candidate rates."""
    a = np.asarray(rate)[...,None]
    variance = a*a*c[:,0,0] - 2*a*c[:,0,1] + c[:,1,1]
    weights = 1/variance
    raw = y-a*x
    offset = np.sum(weights*raw,axis=-1)/np.sum(weights,axis=-1)
    objective = np.sum(weights*(raw-offset[...,None])**2,axis=-1)
    return objective, offset


def calibrate(x, y, c):
    """Grid-bracket every sampled local minimum, refine, reject boundary fits.

    This is a checked numerical search on the declared physical rate interval,
    not a proof of global optimization for arbitrary ill-conditioned datasets.
    """
    x,y,c = validate(x,y,c)
    # Center the coordinates for stable offset elimination.
    mx,my = x.mean(),y.mean(); xc,yc = x-mx,y-my
    grid = np.linspace(*RATE_BOUNDS,65)
    values,_ = profile(grid,xc,yc,c)
    if np.ptp(values) < 1e-10*max(1.,values.max()):
        raise ValueError('Rate direction is unidentifiable')
    candidates=[]
    for i in range(1,len(grid)-1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            fit = minimize_scalar(lambda a: profile(a,xc,yc,c)[0],
                bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-13})
            if fit.success: candidates.append((fit.fun,fit.x))
    if not candidates:
        raise ValueError('No interior rate minimum; inspect bounds and data')
    score,rate = min(candidates)
    if score >= min(values[0],values[-1]) or min(rate-RATE_BOUNDS[0],RATE_BOUNDS[1]-rate)<1e-6:
        raise ValueError('Rate fit reaches physical search boundary')
    _,b = profile(rate,xc,yc,c)
    return np.array([rate,float(b)+my-rate*mx])


def latent_times(x,y,rate,offset,c):
    """Independent per-event Mahalanobis projection onto the fitted line."""
    direction=np.array([1.,rate])
    w=np.linalg.solve(c,np.broadcast_to(direction,(len(c),2))[...,None])[...,0]
    z=np.column_stack((x,np.asarray(y)-offset))
    return np.sum(z*w,axis=1)/(w@direction)


def delay_from_center(center,calibration):
    rate,offset=calibration
    if not np.all(np.isfinite([center,rate,offset])) or rate<=0:
        raise ValueError('Require finite center/offset and positive rate')
    return ((center-ORIGIN)/VELOCITY-offset)/rate-FRAME


def demo(out='outputs',trials=1000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    rng=np.random.default_rng(20261005)
    center=ORIGIN+VELOCITY*(RATE*(FRAME+DELAY)+OFFSET)
    signal=image(center)
    centers=np.array([fit_center(signal+rng.normal(0,READ_SD,len(PIXELS))) for _ in range(trials)])
    draws=rng.normal(size=(trials,len(SYNC),2))
    truth=np.column_stack((SYNC,RATE*SYNC+OFFSET))
    rows=[]
    for h in [1.,2.,4.,8.]:
        c=covariances(h)
        observations=truth+np.einsum('nij,tnj->tni',np.linalg.cholesky(c),draws)
        assumed={'per_event':c,'pooled':np.broadcast_to(c.mean(axis=0),c.shape),
                 'misassigned':np.roll(c,1,axis=0)}
        row={'heterogeneity':h}
        for name,noise in assumed.items():
            estimates=np.array([delay_from_center(centers[i],calibrate(*z.T,noise)) for i,z in enumerate(observations)])
            error=(estimates-DELAY)*1000
            row[name]={'bias_ms':float(error.mean()),'sd_ms':float(error.std(ddof=1)),
                       'rmse_ms':float(np.sqrt(np.mean(error**2))),
                       'mean_error_mcse_ms':float(error.std(ddof=1)/np.sqrt(trials))}
        rows.append(row)
    report={'seed':20261005,'trials_per_level':trials,'events_per_calibration':len(SYNC),
            'truth':{'rate':RATE,'offset_s':OFFSET,'delay_s':DELAY,'frame_s':FRAME},
            'sigma_px':SIGMA,'center_sd_px':float(centers.std(ddof=1)),
            'noiseless_delay_s':float(delay_from_center(center,calibrate(*truth.T,covariances(8)))),
            'sensitivity':rows}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    ax[0].plot(PIXELS,signal-20,'o-');ax[0].set(xlim=(-1,5),xlabel='Sensor coordinate (px)',ylabel='Signal (ADU)',title='Optical spot -> fitted position')
    for c,color,label in zip(covariances(8)[:2],['#197d56','#d35d20'],['High quality','Low quality']):
        vals,vec=np.linalg.eigh(c); angle=np.degrees(np.arctan2(vec[1,-1],vec[0,-1]))
        ax[1].add_patch(Ellipse((0,0),2*np.sqrt(vals[-1]),2*np.sqrt(vals[0]),angle=angle,fill=False,color=color,label=label,lw=2))
    ax[1].set(xlim=(-.27,.27),ylim=(-.09,.09),xlabel='Master timestamp error (s)',ylabel='Reference timestamp error (s)',title='Mahalanobis radius 1; h = 8')
    ax[1].legend(fontsize=8)
    for name,label in [('per_event','Correct per-event ellipses'),('pooled','One mean ellipse'),('misassigned','Ellipses assigned to wrong events')]:
        ax[2].plot([r['heterogeneity'] for r in rows],[r[name]['rmse_ms'] for r in rows],'o-',label=label)
    ax[2].set(xlabel='Low-quality master SD multiplier h',ylabel='Camera-delay RMSE (ms)',title='Same observations, different covariance use');ax[2].legend(fontsize=7)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':demo()
