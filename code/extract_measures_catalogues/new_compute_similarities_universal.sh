#FILES='/local/scratch/ha376/ICM/Imaging_data/digital_pathology_integration/data/catalogues/FFPE_METABRIC_names_catalogs.txt'
#FILES='/home/ICM_CG/Projects/METABRIC/FFPE_METABRIC_names_catalogs_new_Feb2021.txt'
#!/bin/bash
USCOUNTER=0
read -p "Enter name of the dataset: " dataset

if [ "$dataset" == "TCGA" ]
then
    FILES='/home/ICM_CG/Projects/METABRIC/new_data_Ali_May21/tcga_ffpe/TCGA_catalogue_list_025_old.txt'

    while read location id
    do
        echo "Location : $location"
        echo "TCGA.ID : $id"

        python3 new_generate_similarities_universal.py $location $location 10 'TCGA'

      
        USCOUNTER=$(expr $USCOUNTER + 1)
        echo "US counter $USCOUNTER"

    done < <(tail -n "+0" /home/ICM_CG/Projects/METABRIC/new_data_Ali_May21/tcga_ffpe/TCGA_catalogue_list_025_old.txt)
fi


if [ "$dataset" == "METABRIC" ]
then
    FILES='/home/ICM_CG/Projects/METABRIC/FFPE_METABRIC_names_catalogs_new_Feb2021.txt'

    while read id location 
    do
        echo "Location : $location"
        echo "METABRIC.ID : $id"
        
        #python3 new_generate_similarities_universal.py $location $id 10 'METABRIC'
        python3 new_generate_similarities_universal.py $location $USCOUNTER 10 'METABRIC'

      
        USCOUNTER=$(expr $USCOUNTER + 1)
        echo "US counter $USCOUNTER"
    

    done < <(tail -n "+0" /home/ICM_CG/Projects/METABRIC/FFPE_METABRIC_names_catalogs_new_Feb2021.txt)
fi

