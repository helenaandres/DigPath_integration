#!/bin/bash
#for integ in 'Clin+mRNA' 'CNA+mRNA' 'Clin+CNA' 'img_rho+mRNA' 'img_s+mRNA' 'img_v+mRNA'
for integ in  'img_rho+CNA' 'img_s+CNA' 'img_v+CNA' 'img_rho+Clin' 'img_s+Clin' 'img_v+Clin'
#for integ in 'img_s+Clin' 'img_v+Clin'
#for integ in 'Clin+mRNA' 'Clin+CNA' 'CNA+mRNA' 
do
    #for ds in 128
    for ds in 144
    do
        #for lsize in 64
        for lsize in 48
        do
            #for distance in 'kl' 'mmd'
            for distance in 'mmd'
            do
                for beta in 25
                do
                    #for dtype in  'ER' 'DR' 'IC' 'PAM' #'W' whole data 
                    #for dtype in  'total_score' 'tubule.formation' 'lymph_infiltrate' 'nuc_pleomorphism' 'overall_grade' #'W' whole data 
                    for dtype in  'Histological_Type'
                    do
                        for fold in 1 2 3 4 5 #0 whole data
                        do
                            python run_hvae.py --integration=${integ} --ds=${ds} --dtype=${dtype} --fold=${fold} --ls=${lsize} --distance=${distance} --beta=${beta} --writedir='results_k50' --modalities=2
                        done
                    done
                done
            done
        done
    done
done
