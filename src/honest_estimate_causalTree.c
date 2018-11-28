/* 
 * Do honest causalTree estimation with parameters
 * 
 */
 
#include "causalTree.h"
#include "causalTreeproto.h"

    static void
honest_estimate_causalTree0(const int *dimx, int nnode, int nsplit, const int *dimc, 
                            const int *nnum, const int *nodes2, const int *vnum,
                            const double *split2, const int *csplit2, const int *usesur,
                            int *n1, double *wt1, double *dev1, double *yval1, const double *xdata2, 
                            const double *wt2, const double *treatment2, const double *y2,
                            const int *xmiss2, int *where)
{
    Rprintf("honest_estimate_causalTree.c\n");
    int i, j;
    int n;
    int ncat;
    int node, nspl, var, dir;
    int lcount, rcount;
    int npos;
    double temp;
    const int *nodes[3];
    const double *split[4];
    const int **csplit = NULL, **xmiss;
    const double **xdata;
    double *trs = NULL;
    double *cons = NULL; 
    double *trsums = NULL; 
    double *consums = NULL;
    double *trsqrsums = NULL;
    double *consqrsums = NULL;
    int nnodemax = -1;
    int *invertdx = NULL;
    /*add beta*/
    double  y_sum = 0., z_sum = 0.;
    double yz_sum = 0.,  yy_sum = 0., zz_sum = 0.;
   
    
    trs = (double *) ALLOC(nnode, sizeof(double));
    cons = (double *) ALLOC(nnode, sizeof(double));
    trsums = (double *) ALLOC(nnode, sizeof(double));
    consums = (double *) ALLOC(nnode, sizeof(double));
    trsqrsums = (double *) ALLOC(nnode, sizeof(double));
    consqrsums = (double *) ALLOC(nnode, sizeof(double));
    
    /*add beta*/
    y_sum= (double *) ALLOC(nnode, sizeof(double));
    z_sum = (double *) ALLOC(nnode, sizeof(double));
    yz_sum = (double *) ALLOC(nnode, sizeof(double));
    yy_sum = (double *) ALLOC(nnode, sizeof(double));
    zz_sum = (double *) ALLOC(nnode, sizeof(double));
     
    // initialize:
    for (i = 0; i < nnode; i++) {
        trs[i] = 0.;
        cons[i] = 0.;
        trsums[i] = 0.;
        consums[i] = 0.;
        trsqrsums[i] = 0.;
        consqrsums[i] = 0.;
     /* add */
     y_sum[i] = 0.;
     z_sum[i] = 0.;
     yz_sum[i] = 0.;
     yy_sum[i] = 0.;
     zz_sum[i] = 0.;
     
     
        n1[i] = 0;
        wt1[i] = 0.;
        if (nnum[i] > nnodemax) {
            nnodemax = nnum[i]; 
        }
    }
    
    invertdx = (int *) ALLOC(nnodemax + 1, sizeof(int));
    // construct an invert index:
    for (i = 0; i <= nnodemax + 1; i++) {
        invertdx[i] = -1;
    }
    for (i = 0; i != nnode; i++) {
        invertdx[nnum[i]] = i;
    }
    
    n = dimx[0]; // n = # of obs
    for (i = 0; i < 3; i++) {
        nodes[i] = &(nodes2[nnode * i]);
    }
    
    for(i = 0; i < 4; i++) {
        split[i] = &(split2[nsplit * i]);
    }

    if (dimc[1] > 0) {
        csplit = (const int **) ALLOC((int) dimc[1], sizeof(int *));
        for (i = 0; i < dimc[1]; i++)
            csplit[i] = &(csplit2[i * dimc[0]]);
    }
    xmiss = (const int **) ALLOC((int) dimx[1], sizeof(int *));
    xdata = (const double **) ALLOC((int) dimx[1], sizeof(double *));
     Rprintf("The dimx  in honest_estimate_causalTree.c is %d\n", dimx);
    for (i = 0; i < dimx[1]; i++) {
        xmiss[i] = &(xmiss2[i * dimx[0]]);
        xdata[i] = &(xdata2[i * dimx[0]]);
     Rprintf("before current node\n");
    }
    

    for (i = 0; i < n; i++) {
        node = 1;               /* current node of the tree */
next:
        for (npos = 0; nnum[npos] != node; npos++);  /* position of the node */

        n1[npos]++;
        wt1[npos] += wt2[i];
     
     
        trs[npos] += wt2[i] * treatment2[i];
        cons[npos] += wt2[i] * (1 - treatment2[i]);
        /*trsums[npos] += wt2[i] * treatment2[i] * y2[i];*/
      /* add variable*/
        trsums[npos] += wt2[i]  * y2[i];
       
     
        consums[npos] += wt2[i] * (1 - treatment2[i]) * y2[i];
        /*trsqrsums[npos] +=  wt2[i] * treatment2[i] * y2[i] * y2[i];*/
        trsqrsums[npos] +=  wt2[i]  * y2[i] * y2[i];
     
        consqrsums[npos] += wt2[i] * (1 - treatment2[i]) * y2[i] * y2[i];
        
        y_sum[npos] += treatment2[npos];
        z_sum[npos] += y2[npos];
        yz_sum[npos] += y2[npos] * treatment2[npos];
       
        yy_sum[npos] += treatment2[npos] * treatment2[npos];
        zz_sum[npos] += *y2[npos] * *y2[npos];
     
        Rprintf("walk down the tree\n");
        /* walk down the tree */
        nspl = nodes[2][npos] - 1;      /* index of primary split */
        if (nspl >= 0) {        /* not a leaf node */
            var = vnum[nspl] - 1;
            if (xmiss[var][i] == 0) {   /* primary var not missing */
                ncat = (int) split[1][nspl];
                temp = split[3][nspl];
                if (ncat >= 2)
                    dir = csplit[(int) xdata[var][i] - 1][(int) temp - 1];
                else if (xdata[var][i] < temp)
                    dir = ncat;
                else
                    dir = -ncat;
                if (dir) {
                    if (dir == -1)
                        node = 2 * node;
                    else
                        node = 2 * node + 1;
                    goto next;
                }
            }
            if (*usesur > 0) {
                for (j = 0; j < nodes[1][npos]; j++) {
                    nspl = nodes[0][npos] + nodes[2][npos] + j;
                    var = vnum[nspl] - 1;
                    if (xmiss[var][i] == 0) {   /* surrogate not missing */
                        ncat = (int) split[1][nspl];
                        temp = split[3][nspl];
                        if (ncat >= 2)
                            dir = csplit[(int)xdata[var][i] - 1][(int)temp - 1];
                        else if (xdata[var][i] < temp)
                            dir = ncat;
                        else
                            dir = -ncat;
                        if (dir) {
                            if (dir == -1)
                                node = 2 * node;
                            else
                                node = 2 * node + 1;
                            goto next;
                        }
                    }
                }
            }
            if (*usesur > 1) {  /* go with the majority */
                for (j = 0; nnum[j] != (2 * node); j++);
                lcount = n1[j];
                for (j = 0; nnum[j] != (1 + 2 * node); j++);
                rcount = n1[j];
                if (lcount != rcount) {
                    if (lcount > rcount)
                        node = 2 * node;
                    else
                        node = 2 * node + 1;
                    goto next;
                }
            }
        }
        where[i] = node;
    }
    Rprintf("The nnodemax in  honest.causaltree.c is %d\n", nnodemax);
    for (i = 0; i <= nnodemax; i++) {
        if (invertdx[i] == -1)
            continue;
        int origindx = invertdx[i];
     Rprintf("The origindx in  honest.causaltree.c is %d\n", origindx);
        //base case
        /*if (trs[origindx] != 0 && cons[origindx] != 0) {*/
          if (trs[origindx] != 0 && cons[origindx] != 0) {
            /*double tr_mean = trsums[origindx] * 1.0 / trs[origindx];*/
            double tr_mean = trsums[origindx] * 1.0 / wt1[origindx];
            double con_mean = consums[origindx] * 1.0 / cons[origindx];
            double tt_mean =  yy_sum[origindx]* 1.0 / wt1[origindx];
           
            /*yval1[origindx] = tr_mean - con_mean;*/
            /*dev1[origindx] = trsqrsums[origindx] - trs[origindx] * tr_mean * tr_mean 
                + consqrsums[origindx] - cons[origindx] * con_mean * con_mean;*/
           yval1[origindx] =  (wt1[origindx] * yz_sum[origindx] - z_sum[origindx] * y_sum[origindx]) / (wt1[origindx] * yy_sum[origindx] - y_sum[origindx] * y_sum[origindx]);
           dev1[origindx] = yy_sum[origindx] -  tt_mean * tt_mean;
           
           Rprintf("The trsqrsums in  honest.causaltree.c is %d\n", trsqrsums);
           Rprintf("The consqrsums in  honest.causaltree.c is %d\n", consqrsums);
           
        } else {
            int parentdx = invertdx[i / 2];
            yval1[origindx] = yval1[parentdx];
            dev1[origindx] = yval1[parentdx];
         
        }
    Rprintf("The dev1 in honest.causaltree.c is %d\n", dev1);
    Rprintf("The yval1 in honest.causaltree.c is %d\n", yval1);
    }
    Rprintf("end the honest estimate tree\n");
    
}
   
#include <Rinternals.h>

SEXP
honest_estimate_causalTree(SEXP dimx, SEXP nnode, 
                           SEXP nsplit, SEXP dimc, SEXP nnum, 
                           SEXP nodes2, 
                           SEXP n1, SEXP wt1, SEXP dev1, SEXP yval1, 
                           SEXP vnum, 
                           SEXP split2,
                           SEXP csplit2, SEXP usesur, 
                           SEXP xdata2, SEXP wt2, SEXP treatment2, SEXP y2,
                           SEXP xmiss2)
{
    int n = asInteger(dimx);
    SEXP where = PROTECT(allocVector(INTSXP, n));
    honest_estimate_causalTree0(INTEGER(dimx), asInteger(nnode), asInteger(nsplit),
            INTEGER(dimc), INTEGER(nnum), INTEGER(nodes2),
            INTEGER(vnum), REAL(split2), INTEGER(csplit2),
            INTEGER(usesur), 
            INTEGER(n1), REAL(wt1), REAL(dev1), REAL(yval1), 
            REAL(xdata2), REAL(wt2), REAL(treatment2), REAL(y2),
            INTEGER(xmiss2), INTEGER(where));
    UNPROTECT(1);
    return where;
}
