/*
 * split.Rule = CT
 */
#include <math.h>
#include "causalTree.h"
#include "causalTreeproto.h"

static double *sums, *wtsums, *treatment_effect;
static double *wts, *trs, *trsums;
static int *countn;
static int *tsplit;
static double *wtsqrsums, *trsqrsums;

/*categorical var*/
static double *y_, *z_ , *yz_ ,  *yy_ , *zz_ ;



int
CTinit(int n, double *y[], int maxcat, char **error,
        int *size, int who, double *wt, double *treatment, 
        int bucketnum, int bucketMax, double *train_to_est_ratio)
{
    if (who == 1 && maxcat > 0) {
        graycode_init0(maxcat);
        countn = (int *) ALLOC(2 * maxcat, sizeof(int));
        tsplit = countn + maxcat;
        treatment_effect = (double *) ALLOC(8 * maxcat, sizeof(double));
        wts = treatment_effect + maxcat;
        trs = wts + maxcat;
        sums = trs + maxcat;
        wtsums = sums + maxcat;
        trsums = wtsums + maxcat;
        wtsqrsums = trsums + maxcat;
        trsqrsums = wtsqrsums + maxcat;
        
        y_ = (double *) ALLOC(5 * maxcat, sizeof(double));
        z_ = y_ + maxcat;
        yz_ = z_ + maxcat;
        yy_ = yz_ + maxcat;
        zz_ = yy_ + maxcat;
    }
    *size = 1;
    *train_to_est_ratio = n * 1.0 / ct.NumHonest;
    return 0;
}

void
CTss(int n, double *y[], double *value,  double *con_mean, double *tr_mean, 
     double *risk, double *wt, double *treatment, double max_y,
     double alpha, double train_to_est_ratio)
{
    int i;
    double temp0 = 0., temp1 = 0., twt = 0.; /* sum of the weights */ 
    double ttreat = 0.;
    double effect;
    double tr_var, con_var;
    double con_sqr_sum = 0., tr_sqr_sum = 0.;
    double var_beta = 0., beta_sqr_sum = 0.; /* var */
    double  y_sum = 0., z_sum = 0.;
    double yz_sum = 0.,  yy_sum = 0., zz_sum = 0.;
    double  beta_1 = 0.;
    double beta_0 = 0.;    
    
    double z_hat_sum=0.; 
        
    for (i = 0; i < n; i++) {
        temp1 += *y[i] * wt[i] * treatment[i];
        temp0 += *y[i] * wt[i] * (1 - treatment[i]);
        twt += wt[i];
        ttreat += wt[i] * treatment[i];
        tr_sqr_sum += (*y[i]) * (*y[i]) * wt[i] * treatment[i];
        con_sqr_sum += (*y[i]) * (*y[i]) * wt[i] * (1- treatment[i]);
        
        y_sum += treatment[i];
        z_sum += *y[i];
        yz_sum += *y[i] * treatment[i];
       
        yy_sum += treatment[i] * treatment[i];
        zz_sum += *y[i] * *y[i];
        //z_hat_sum += (*y[i]-beta_0-beta_1*treatment[i])* (*y[i]-beta_0-beta_1*treatment[i]);
    }

   
    //effect = temp1 / ttreat - temp0 / (twt - ttreat);        
    tr_var = tr_sqr_sum / ttreat - temp1 * temp1 / (ttreat * ttreat);
    con_var = con_sqr_sum / (twt - ttreat) - temp0 * temp0 / ((twt - ttreat) * (twt - ttreat));
   
   
    /* Y= beta_0 + beta_1 treatment , ONLY one pair*/
   
    for (i = 0; i < n; i++) {
      z_hat_sum += (*y[i]-beta_0-beta_1*treatment[i])* (*y[i]-beta_0-beta_1*treatment[i]);
    }
     
    beta_1 = (twt * yz_sum - z_sum * y_sum) / (twt * yy_sum - y_sum * y_sum); 
    beta_0 = (z_sum - beta_1 * y_sum) / twt;
    effect = beta_1;
    beta_sqr_sum = beta_1 * beta_1 ;
        
    var_beta = (z_hat_sum/(twt-2))/(yy_sum - y_sum * y_sum/twt) ;
        
    *tr_mean = temp1 / ttreat;
    *con_mean = temp0 / (twt - ttreat);
    *value = effect;
        
    //*risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + 
    //(1 - alpha) * (1 + train_to_est_ratio) * twt * (tr_var /ttreat  + con_var / (twt - ttreat));
    *risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + (1 - alpha) * (1 + train_to_est_ratio) * twt * ( var_beta);
    
 }





void CT(int n, double *y[], double *x, int nclass, int edge, double *improve, double *split, 
        int *csplit, double myrisk, double *wt, double *treatment,  int minsize, double alpha,
        double train_to_est_ratio)
{
          
    int i, j;
    double temp;
    double left_sum, right_sum;
    double left_tr_sum, right_tr_sum;
    double left_tr, right_tr;
    double left_wt, right_wt;
    int left_n, right_n;
    double best;
    int direction = LEFT;
    int where = 0;
    double node_effect, left_effect, right_effect;
    double left_temp, right_temp;
    int min_node_size = minsize;
    
    double tr_var, con_var;
    double right_sqr_sum, right_tr_sqr_sum, left_sqr_sum, left_tr_sqr_sum;
    double left_tr_var, left_con_var, right_tr_var, right_con_var;

    right_wt = 0.;
    right_tr = 0.;
    right_sum = 0.;
    right_tr_sum = 0.;
    right_sqr_sum = 0.;
    right_tr_sqr_sum = 0.;
    right_n = n;
    double   right_y_sum = 0., right_z_sum = 0.;
    double  left_y_sum = 0., left_z_sum = 0.;
    double right_yz_sum = 0.,  right_yy_sum = 0., right_zz_sum = 0.;
    double left_yz_sum = 0.,  left_yy_sum = 0., left_zz_sum = 0.;
    double  beta_1 = 0., beta_0 = 0.;
    
    double   beta_sqr_sum = 0.,  var_beta = 0.; /* beta*/
    
        
    double  twt = 0.; 
    double  y_sum = 0., z_sum = 0.;
    double yz_sum = 0.,  yy_sum = 0., zz_sum = 0.;
    double z_hat_sum=0.;
        
    for (i = 0; i < n; i++) {
        right_wt += wt[i];
        right_tr += wt[i] * treatment[i];
        right_sum += *y[i] * wt[i];
        right_tr_sum += *y[i] * wt[i] * treatment[i];
        right_sqr_sum += (*y[i]) * (*y[i]) * wt[i];
        right_tr_sqr_sum += (*y[i]) * (*y[i]) * wt[i] * treatment[i];
      
       
        right_y_sum += treatment[i];
        right_z_sum += *y[i];
        right_yz_sum += *y[i] * treatment[i];
       
        right_yy_sum += treatment[i] * treatment[i];
        right_zz_sum += *y[i] * *y[i];
    }

    //Rprintf("The nclass in function CT in CT.c is %d\n",(int)nclass);  
        
        
    beta_1 = (right_wt * right_yz_sum - right_z_sum * right_y_sum) / (right_wt * right_yy_sum - right_y_sum * right_y_sum);
    beta_0 = (right_z_sum - beta_1 * right_y_sum) / right_wt;
    
    beta_sqr_sum = beta_1 * beta_1 ;
       
    for (i = 0; i < n; i++) {
      z_hat_sum += (*y[i]-beta_0-beta_1*treatment[i]) * (*y[i]-beta_0-beta_1*treatment[i]);
    }
     
        
    var_beta = (z_hat_sum/(right_wt-2)) / (right_yy_sum-right_y_sum * right_y_sum/right_wt) ;
        
    temp = beta_1;  
        
    //var_beta = beta_sqr_sum / right_wt - beta_1 * beta_1 / (right_wt * right_wt);
    //temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) / (right_wt - right_tr);
    tr_var = right_tr_sqr_sum / right_tr - right_tr_sum * right_tr_sum / (right_tr * right_tr);
    con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
        - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
        / ((right_wt - right_tr) * (right_wt - right_tr));
   /* node_effect = alpha * temp * temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
        * right_wt * (tr_var / right_tr  + con_var / (right_wt - right_tr));*/
  
    node_effect = alpha * temp * temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
        * right_wt * (var_beta);
   
     
        
    
    if (nclass == 0) {
        /* continuous predictor */
        left_wt = 0;
        left_tr = 0;
        left_n = 0;
        left_sum = 0;
        left_tr_sum = 0;
        left_sqr_sum = 0;
        left_tr_sqr_sum = 0;
        best = 0;
        
        for (i = 0; right_n > edge; i++) {
       
            left_wt += wt[i];
            right_wt -= wt[i];
            left_tr += wt[i] * treatment[i];
            right_tr -= wt[i] * treatment[i];
            left_n++;
            right_n--;
            temp = *y[i] * wt[i] * treatment[i];
            left_tr_sum += temp;
            right_tr_sum -= temp;
            left_sum += *y[i] * wt[i];
            right_sum -= *y[i] * wt[i];
            temp = (*y[i]) *  (*y[i]) * wt[i];
            left_sqr_sum += temp;
            right_sqr_sum -= temp;
            temp = (*y[i]) * (*y[i]) * wt[i] * treatment[i];
            left_tr_sqr_sum += temp;
            right_tr_sqr_sum -= temp;
                
           
            left_y_sum += treatment[i];
            right_y_sum -= treatment[i];
            left_z_sum += *y[i];
            right_z_sum -= *y[i];
            left_yz_sum += *y[i] * treatment[i];
            right_yz_sum -= *y[i] * treatment[i];
           
            left_yy_sum += treatment[i] * treatment[i];
            right_yy_sum -= treatment[i] * treatment[i];
            left_zz_sum += *y[i] * *y[i];
            right_zz_sum -= *y[i] * *y[i];
            
                /* change treatment */
           /* if (x[i + 1] != x[i] && left_n >= edge &&
                (int) left_tr >= min_node_size &&
                (int) left_wt - (int) left_tr >= min_node_size &&
                (int) right_tr >= min_node_size &&
                (int) right_wt - (int) right_tr >= min_node_size) {  */                           
                                            
               if (x[i + 1] != x[i] && left_n >= edge &&
                (int) left_wt >= min_node_size &&
                (int) right_wt  >= min_node_size) {
                     
    beta_1 = (left_wt * left_yz_sum - left_z_sum * left_y_sum) / (left_wt * left_yy_sum - left_y_sum * left_y_sum);
    beta_0 = (left_z_sum - beta_1 * left_y_sum) / left_wt;
    
    beta_sqr_sum = beta_1 * beta_1 ;
    
               
    //z_hat_sum += (*y[i]-beta_0-beta_1*treatment[i]) * (*y[i]-beta_0-beta_1*treatment[i]);
     
                       
    var_beta = (z_hat_sum/(left_wt-2)) / (left_yy_sum -left_y_sum * left_y_sum/left_wt) ;
                 
    //var_beta = beta_sqr_sum / left_wt - beta_1 * beta_1 / (left_wt * left_wt);
    
    left_temp = beta_1;
    left_effect = left_temp * left_temp * left_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                    * left_wt * (var_beta);
                   
      //left_temp = left_tr_sum / left_tr - (left_sum - left_tr_sum) / (left_wt - left_tr);
                /*left_tr_var = left_tr_sqr_sum / left_tr - 
                    left_tr_sum  * left_tr_sum / (left_tr * left_tr);
                left_con_var = (left_sqr_sum - left_tr_sqr_sum) / (left_wt - left_tr)  
                    - (left_sum - left_tr_sum) * (left_sum - left_tr_sum)
                    / ((left_wt - left_tr) * (left_wt - left_tr));        
                left_effect = alpha * left_temp * left_temp * left_wt
                        - (1 - alpha) * (1 + train_to_est_ratio) * left_wt 
                    * (left_tr_var / left_tr + left_con_var / (left_wt - left_tr));*/         

    beta_1 = (right_wt * right_yz_sum - right_z_sum * right_y_sum) / (right_wt * right_yy_sum - right_y_sum * right_y_sum);
    beta_0 = (right_z_sum - beta_1 * right_y_sum) / right_wt;
    
    beta_sqr_sum = beta_1 * beta_1 ;
                       
   // z_hat_sum += (*y[i]-beta_0-beta_1*treatment[i]) * (*y[i]-beta_0-beta_1*treatment[i]);
    
    var_beta = (z_hat_sum/(right_wt-2)) / (right_yy_sum -right_y_sum * right_y_sum/right_wt) ;
                       
    //var_beta = beta_sqr_sum / right_wt - beta_1 * beta_1 / (right_wt * right_wt);
    
    right_temp = beta_1;
    right_effect = right_temp * right_temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                    * right_wt * (var_beta);
                    
//right_temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) / (right_wt - right_tr);
                /*right_tr_var = right_tr_sqr_sum / right_tr -
                    right_tr_sum * right_tr_sum / (right_tr * right_tr);
                right_con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
                    - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
                    / ((right_wt - right_tr) * (right_wt - right_tr));
                right_effect = alpha * right_temp * right_temp * right_wt
                        - (1 - alpha) * (1 + train_to_est_ratio) * right_wt * 
                            (right_tr_var / right_tr + right_con_var / (right_wt - right_tr));*/
                
                temp = left_effect + right_effect - node_effect;
                       /*check beta*/
                       
           
                       
                if (temp>best) {Rprintf("cont: compare temp and best\n");
                    best = temp;
                    where = i;     
                                  Rprintf("best after in cont %d.\n", best);
                    if (left_temp < right_temp){
                        direction = LEFT;
                    }
                    else{
                        direction = RIGHT;
                    }
                }             
            }
        }
        
        *improve = best;
        if (best > 0) {         /* found something */
        csplit[0] = direction;
            *split = (x[where] + x[where + 1]) / 2; 
        }
    }
    
    /*
    * Categorical predictor
    */
    else {
        Rprintf("come in Categorical predictor\n");
        for (i = 0; i < nclass; i++) {
            countn[i] = 0;
            wts[i] = 0;
            trs[i] = 0;
            sums[i] = 0;
            wtsums[i] = 0;
            trsums[i] = 0;
            wtsqrsums[i] = 0;
            trsqrsums[i] = 0;
                
        y_[i] =  0;
        z_ [i]=  0;
        yz_[i] =  0;
        yy_[i] =  0;
        zz_ [i]=  0;
                
        }
        
        /* rank the classes by treatment effect */
        for (i = 0; i < n; i++) {
            j = (int) x[i] - 1;
            countn[j]++;
            wts[j] += wt[i];
            trs[j] += wt[i] * treatment[i];
            sums[j] += *y[i];
            wtsums[j] += *y[i] * wt[i];
            trsums[j] += *y[i] * wt[i] * treatment[i];
            wtsqrsums[j] += (*y[i]) * (*y[i]) * wt[i];
            trsqrsums[j] +=  (*y[i]) * (*y[i]) * wt[i] * treatment[i];
            
        y_[j] += treatment[i];
        z_[j] += *y[i];   
        yz_[j] += *y[i] * treatment[i];
        yy_[j] += treatment[i] * treatment[i];
        zz_[j] += *y[i] * *y[i];
       
            
        }
        
	Rprintf("nclass in function CT in CT.c is %d\n", nclass);
        for (i = 0; i < nclass; i++) {
		Rprintf("i in nclass in function CT in CT.c is %d\n", i);
		
            if (countn[i] > 0) {
		    
                tsplit[i] = RIGHT;   
               treatment_effect[i] = (wts[i] * yz_[i] - z_[i] * y_[i]) / (wts[i] * yy_[i] - y_[i] * y_[i]);
                /*treatment_effect[i] = ( twt[j] * yz_sum[j] - z_sum[j] * y_sum[j]) / (twt[j] * yy_sum[j] - y_sum[j] * y_sum[j]); */
           Rprintf("treatment_effect[i] in function CT in CT.c is %d\n", treatment_effect[i]);  
            } else
                tsplit[i] = 0;
                
        }
            
        graycode_init2(nclass, countn, treatment_effect);
            
        
        /*
         * Now find the split that we want
         */
        
        left_wt = 0;
        left_tr = 0;
        left_n = 0;
        left_sum = 0;
        left_tr_sum = 0;
        left_sqr_sum = 0.;
        left_tr_sqr_sum = 0.;
        
        best = 0;
        where = 0;
        while ((j = graycode()) < nclass) {
            tsplit[j] = LEFT;
            left_n += countn[j];
            right_n -= countn[j];
            
            left_wt += wts[j];
            right_wt -= wts[j];
            
            left_tr += trs[j];
            right_tr -= trs[j];
            
            left_sum += wtsums[j];
            right_sum -= wtsums[j];
            
            left_tr_sum += trsums[j];
            right_tr_sum -= trsums[j];
            
            left_sqr_sum += wtsqrsums[j];
            right_sqr_sum -= wtsqrsums[j];
            
            left_tr_sqr_sum += trsqrsums[j];
            right_tr_sqr_sum -= trsqrsums[j];
            
                
            left_y_sum += y_[j];
            right_y_sum -= y_[j];
            left_z_sum += z_[j];
            right_z_sum -= z_[j];
            left_yz_sum += yz_[j];
            right_yz_sum -= yz_[j];
           
            left_yy_sum += yy_[j];
            right_yy_sum -= yy_[j];
            left_zz_sum += zz_[j];
            right_zz_sum -= zz_[j];
                
                
                
            if (left_n >= edge && right_n >= edge &&
                
                (int) left_wt  >= min_node_size &&
                
                (int) right_wt >= min_node_size) {
                
                    
               /* change treatment split*/
    beta_1 = (left_n * left_yz_sum - left_z_sum * left_y_sum) / (left_n * left_yy_sum - left_y_sum * left_y_sum);
    beta_0 = (left_z_sum - beta_1 * left_y_sum) / left_n;
    left_temp = beta_0;
    beta_sqr_sum = beta_1 * beta_1 ;
    
                    Rprintf("beta_1 in cat in CT.c %d.\n", beta_1);  
                    
                    
   /* for (i = 0; i < nclass; i++) {
    z_hat_sum += (*y[i]-beta_0-beta_1*treatment[i]) * (*y[i]-beta_0-beta_1*treatment[i]);
    }
    */
     
    Rprintf("z_hat_sum in cat in CT.c %d.\n", z_hat_sum);    
                    
                    
                    
    var_beta = (z_hat_sum/(left_n-2)) / (left_yy_sum -left_y_sum * left_y_sum/left_n) ;
               
    //var_beta = beta_sqr_sum / n - beta_1 * beta_1 / (n * n);
    
   
    left_effect = left_temp * left_temp * left_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                    * left_wt * (var_beta);

                    
                    
                    
                    
    beta_1 = (right_n * right_yz_sum - right_z_sum * right_y_sum) / (right_n * right_yy_sum - right_y_sum * right_y_sum);
    beta_0 = (right_z_sum - beta_1 * right_y_sum) / right_n;
    right_temp = beta_0;
    beta_sqr_sum = beta_1 * beta_1 ;
    
    /*for (i = 0; i < nclass; i++) {
    z_hat_sum += (*y[i]-beta_0-beta_1*treatment[i]) * (*y[i]-beta_0-beta_1*treatment[i]);
    }*/
    
                 
    var_beta = (z_hat_sum/(right_n-1)) / (right_yy_sum  -right_y_sum * right_y_sum/  right_n) ;
               
    //var_beta = beta_sqr_sum / n - beta_1 * beta_1 / (n * n);
    
   
    right_effect = right_temp * right_temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                    * right_wt * (var_beta);
                    
                    
                    
                /*left_temp = left_tr_sum / left_tr - (left_sum - left_tr_sum) 
                    / (left_wt - left_tr); 
         
                        
                left_tr_var = left_tr_sqr_sum / left_tr 
                    - left_tr_sum  * left_tr_sum / (left_tr * left_tr);
                
                left_con_var = (left_sqr_sum - left_tr_sqr_sum) / (left_wt - left_tr)  
                    - (left_sum - left_tr_sum) * (left_sum - left_tr_sum)
                    / ((left_wt - left_tr) * (left_wt - left_tr));       
                
                left_effect = alpha * left_temp * left_temp * left_wt
                    - (1 - alpha) * (1 + train_to_est_ratio) * left_wt * 
                        (left_tr_var / left_tr + left_con_var / (left_wt - left_tr));
                
                right_temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) 
                    / (right_wt - right_tr);
                right_tr_var = right_tr_sqr_sum / right_tr 
                    - right_tr_sum * right_tr_sum / (right_tr * right_tr);
                right_con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
                    - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
                    / ((right_wt - right_tr) * (right_wt - right_tr));
                right_effect = alpha * right_temp * right_temp * right_wt
                        - (1 - alpha) * (1 + train_to_est_ratio) * right_wt *
                            (right_tr_var / right_tr + right_con_var / (right_wt - right_tr));*/
                    
                temp = left_effect + right_effect - node_effect;
		    
                 Rprintf("left_effect in cat in CT.c %d.\n", left_effect); 
                Rprintf("right_effect in cat in CT.c %d.\n", right_effect); 
		 Rprintf("node_effect in cat in CT.c %d.\n", node_effect); 
		    
                Rprintf("temp in cat in CT.c %d.\n", temp); 
                Rprintf("best in cat in CT.c %d.\n", best); 
               Rprintf("compare in cat in CT.c %d.\n", temp>0);
		    
      if (temp > best) {
		    Rprintf("YES!cat: compare temp and best\n");
                    best = temp;
				  
                    Rprintf("best after in cat is %d\n", best);
				  
                    if (left_temp > right_temp)
                        for (i = 0; i < nclass; i++) csplit[i] = -tsplit[i];
                    else
                        for (i = 0; i < nclass; i++) csplit[i] = tsplit[i];
		}
	    }
	}

        *improve = best;
	    Rprintf("improve cat is %d\n", *improve);
    }
        Rprintf("End function CT in CT.c \n");
} /*CT FUNCTION*/


double
    CTpred(double *y, double wt, double treatment, double *yhat, double propensity)
    {
        double ystar;
        double temp;
        
        ystar = y[0] * (treatment - propensity) / (propensity * (1 - propensity));
        temp = ystar - *yhat;
        return temp * temp * wt;
    }
