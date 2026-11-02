// Independent scalar exhaustive MPC audit. No simulator or producer calls.
#include <algorithm>
#include <cmath>
#include <limits>

static double compensated_sum(const double* values, int count) {
    double partial[32]; int n = 0;
    for (int k = 0; k < count; ++k) {
        double x = values[k]; int used = 0;
        for (int j = 0; j < n; ++j) {
            double y = partial[j];
            if (std::abs(x) < std::abs(y)) std::swap(x, y);
            double hi = x + y, lo = y - (hi - x);
            if (lo != 0.0) partial[used++] = lo;
            x = hi;
        }
        n = used; partial[n++] = x;
    }
    double hi = 0.0, lo = 0.0;
    if (n) {
        hi = partial[--n];
        while (n) {
            double x = hi, y = partial[--n];
            hi = x + y; lo = y - (hi - x);
            if (lo != 0.0) break;
        }
        if (n && ((lo < 0 && partial[n-1] < 0) || (lo > 0 && partial[n-1] > 0))) {
            double y = 2*lo, x = hi+y;
            if (x-hi == y) hi = x;
        }
    }
    return hi;
}

struct Search {
    const double *mean, *scale, *coef, *reference;
    double envelope, best; int horizon;
    long long valid=0, nonfinite=0, unsupported=0;
    double current[8]{}, chosen[8]{};
    static long long leaves(int depth) { long long n=1; while (depth--) n*=3; return n; }
    bool step(double q, double v, double m, double u, double& nq, double& nv, double& nm) {
        nm=m;
        for (int k=0;k<25;++k) nm=std::max(-1.0,std::min(1.0,nm-0.01*u));
        double raw[4]={q,m,nm,v}, z[4], x[15], terms[15], residual[2];
        for (int k=0;k<4;++k) x[k]=z[k]=(raw[k]-mean[k])/scale[k];
        int k=4;
        for (int i=0;i<4;++i) for (int j=i;j<4;++j) x[k++]=z[i]*z[j];
        x[14]=1.0;
        for (int col=0;col<2;++col) {
            for (int i=0;i<15;++i) terms[i]=x[i]*coef[2*i+col];
            residual[col]=compensated_sum(terms,15);
        }
        double rawq=q+v+residual[0]; nv=v+residual[1];
        if (!std::isfinite(rawq) || !std::isfinite(nv)) return false;
        nq=std::max(0.0,std::min(1.0,rawq)); return true;
    }
    void visit(int depth,double q,double v,double m,double cost) {
        if (depth==horizon) {
            ++valid;
            if (cost<best) { best=cost; std::copy(current,current+horizon,chosen); }
            return;
        }
        const double actions[3]={0.0,-1.0,1.0};
        for (double u:actions) {
            double nq,nv,nm;
            if (!step(q,v,m,u,nq,nv,nm)) { nonfinite+=leaves(horizon-depth-1); continue; }
            if (std::abs(nv)>envelope) { unsupported+=leaves(horizon-depth-1); continue; }
            current[depth]=u;
            double error=nq-reference[depth];
            visit(depth+1,nq,nv,nm,cost+error*error+0.0001*u*u);
        }
    }
};

extern "C" int velocity_enumerate(const double* mean,const double* scale,const double* coef,
 double envelope,double q,double v,double motor,const double* reference,int horizon,
 long long* counts,double* best,double* sequence) {
    if (horizon<1 || horizon>8 || !std::isfinite(envelope) || envelope<=0) return 1;
    for (int i=0;i<4;++i) if (!std::isfinite(scale[i]) || scale[i]<=0) return 2;
    Search search{mean,scale,coef,reference,envelope,std::numeric_limits<double>::infinity(),horizon};
    search.visit(0,q,v,motor,0.0);
    counts[0]=search.valid; counts[1]=search.nonfinite; counts[2]=search.unsupported;
    *best=search.best; std::copy(search.chosen,search.chosen+horizon,sequence); return 0;
}
