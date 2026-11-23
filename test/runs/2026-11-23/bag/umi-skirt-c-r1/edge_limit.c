#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Project native node coordinates before the solver sees the next accepted
   state. A negative mass is never used; zero mobility denotes a gripped node. */
double project_edges(double *v, const int64_t *edges, const double *rest,
                     const double *mobility, size_t count) {
    double peak = 0;
    for (int sweep=0; sweep<=30; ++sweep) {
        peak = 0;
        for (size_t i=0; i<count; ++i) {
            const int64_t a=edges[2*i], b=edges[2*i+1];
            double *x=v+3*a, *y=v+3*b;
            const double dx=y[0]-x[0], dy=y[1]-x[1], dz=y[2]-x[2];
            const double length=sqrt(dx*dx+dy*dy+dz*dz), ratio=length/rest[i];
            if (ratio>peak) peak=ratio;
            if (sweep==0 || length<=1.03*rest[i]) continue;
            const double mobility_sum=mobility[a]+mobility[b];
            if (mobility_sum==0) continue;
            const double f=(length-1.03*rest[i])/(length*mobility_sum);
            const double del[3]={dx,dy,dz};
            for (int k=0;k<3;k++) {
                x[k]+=f*mobility[a]*del[k]; y[k]-=f*mobility[b]*del[k];
            }
        }
        if (peak<=1.045) break;
    }
    return peak;
}
