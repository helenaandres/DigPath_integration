#FILES='/local/scratch/ha376/ICM/Imaging_data/digital_pathology_integration/data/catalogues/FFPE_METABRIC_names_catalogs.txt'
FILES='/home/ICM_CG/Projects/METABRIC/FFPE_METABRIC_names_catalogs_new_Feb2021.txt'
USCOUNTER=0

python3 add_header.py '46822.svs' 'MB-0000' 5

while read id location 
do
    echo "Location : $location"
    echo "METABRIC.ID : $id"
        
    python3 generate_similarities.py $location $id 10

      
    USCOUNTER=$(expr $USCOUNTER + 1)
    echo "US counter $USCOUNTER"
    
#done < <(tail -n "+200" ./FFPE_METABRIC_names_catal_0.txt)
#done < <(tail -n "+0" /home/ICM_CG/Projects/METABRIC/FFPE_METABRIC_names_catalogs2.txt)
done < <(tail -n "+0" /home/ICM_CG/Projects/METABRIC/FFPE_METABRIC_names_catalogs_new_Feb2021.txt)
