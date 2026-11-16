# Clinical_Frequency.py

import io
import os
import numpy as np
import pandas as pd
import requests
import json
import time
import textwrap
import re

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from pandas import json_normalize


# ------------------------------------------------------------
# Download latest ClinVar data
# ------------------------------------------------------------

clinvar_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
output_file = "variation_summary.txt.gz"

print("Downloading ClinVar variant summary...")

response = requests.get(
    clinvar_url,
    stream=True,
    timeout=120
)
response.raise_for_status()

with open(output_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)

print("Download complete.")


# ------------------------------------------------------------
# Read ClinVar data
# ------------------------------------------------------------

print("Reading ClinVar data...")

clinvar = pd.read_csv(
    output_file,
    sep="\t",
    compression="gzip",
    low_memory=False
)

clinvar = clinvar.convert_dtypes()

print(f"Loaded {len(clinvar):,} ClinVar rows.")

## Trim for efficiency
pathovar = clinvar[clinvar['ClinicalSignificance'].str.contains('Pathogenic', case=False, na=False)]
pathovar = pathovar[pathovar['Assembly'].str.contains('GRCh38', na=False)]

pathovar = pathovar.drop(['#AlleleID', 
                    'Type',
                    'Name',
                    'HGNC_ID',
                    'nsv/esv (dbVar)',
                    'RCVaccession',
                    'PhenotypeIDS',
                    'Origin',
                    'Start',
                    'Stop',
                    'ReferenceAllele',
                    'AlternateAllele',
                    'Cytogenetic',
                    'ChromosomeAccession',
                    'Guidelines',
                    'OtherIDs',
                    'NumberSubmitters',
                    'ClinSigSimple',
                    'GeneID',
                    'SubmitterCategories',
                    'VariationID',
                    'ReviewStatusOncogenicity',
                    'SCVsForAggregateGermlineClassification',
                    'SCVsForAggregateSomaticClinicalImpact',
                    'SCVsForAggregateOncogenicityClassification', 
                    'TestedInGTR'], axis=1)


## Create new lines for each item, and filter

pathovar['GeneSymbol'] = pathovar['GeneSymbol'].str.split(';')
pathovar = pathovar.explode('GeneSymbol')

pathovar['PhenotypeList'] = pathovar['PhenotypeList'].str.split('|')
pathovar = pathovar.explode('PhenotypeList')

pathovar['PhenotypeList'] = pathovar['PhenotypeList'].str.split(';')
pathovar = pathovar.explode('PhenotypeList')

pathovar['ClinicalSignificance'] = pathovar['ClinicalSignificance'].str.split('/')
pathovar = pathovar.explode('ClinicalSignificance')

pathovar['ClinicalSignificance'] = pathovar['ClinicalSignificance'].str.split(';')
pathovar = pathovar.explode('ClinicalSignificance')

### Only takes pathogenic variants, need a second version for including likely pathogenic
pathovar = pathovar[
    pathovar['ClinicalSignificance'].str.contains(
        'Pathogenic',
        case=False,
        na=False
    )
]

pathovar = pathovar[
    ~pathovar['ClinicalSignificance'].str.contains(
        'conflicting',
        case=False,
        na=False
    )
]

print("Rows before removing duplicates:", len(pathovar))

pathovar = pathovar.drop_duplicates(
    subset=[
        'GeneSymbol',
        'PhenotypeList',
        'Chromosome',
        'PositionVCF',
        'ReferenceAlleleVCF',
        'AlternateAlleleVCF'
    ]
)

print("Rows after removing duplicates:", len(pathovar))

print(
    "Unique variants:",
    pathovar[
        [
            'Chromosome',
            'PositionVCF',
            'ReferenceAlleleVCF',
            'AlternateAlleleVCF'
        ]
    ].drop_duplicates().shape[0]
)

pathovar = pathovar[
    ~pathovar['PhenotypeList'].str.contains(
        'conditions|See cases|not provided|not specified',
        na=False
    )
]

pathovar = pathovar.reset_index(drop=True)


# here


## Additionally parse for phenotypes of interest too

#pathovar = pathovar[pathovar['PhenotypeList'].str.contains('cancer|neoplasm|tumor|tumour|carcinoma|astoma|cytoma|neoplasia|cystoma|artoma', case=False, na=False)]

pathovar.to_csv("Pathogenic_Variants_temp.csv", index=False)


# In[3]:



## Create Variant List from ClinVar Data

pvars = pd.read_csv("Pathogenic_Variants_temp.csv", delimiter=",")
pvars['variant'] = pvars['Chromosome'].astype(str) + '-' + pvars['PositionVCF'].astype(str) + '-' + pvars['ReferenceAlleleVCF'] + '-' + pvars['AlternateAlleleVCF']

pvars.to_csv("Pathogenic_Variants.csv", index=False)


main = "17-43045711-G-GT"

## Search by variant for allele frequencies from GnomAD
### Run in background

gnomadurl = "https://gnomad.broadinstitute.org/api?"
gnomadvar= "query=%20%20%20%20%7B%0A%09%09variant(variantId%3A%20%22" + main + "%22%2C%20dataset%3A%20gnomad_r4)%20%7B%0A%20%20%20%20%20%20%20%20variant_id%0A%20%20%20%20%20%20%20%20transcript_consequences%20%7B%0A%20%20%20%20%20%20%20%20%20%20gene_symbol%0A%20%20%20%20%20%20%20%20%09gene_id%0A%20%20%20%20%20%20%20%20%20%20is_canonical%0A%20%20%20%20%20%20%20%20%20%20is_mane_select%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%09rsid%0A%20%20%20%20%20%20%09in_silico_predictors%20%7B%0A%20%20%20%20%20%20%09%20%20id%0A%20%20%20%20%20%20%20%20%20%20value%0A%20%20%20%20%20%20%09%7D%0A%20%20%20%20%20%20%20%20genome%20%7B%0A%20%20%20%20%20%20%09%09populations%20%7B%0A%20%20%20%20%20%20%09%09%20%20id%0A%20%20%20%20%20%20%20%20%20%20%20%20ac%0A%20%20%20%20%20%20%20%20%20%20%20%20an%0A%20%20%20%20%20%20%20%20%20%20%20%20homozygote_count%0A%20%20%20%20%20%20%20%20%20%20%20%20hemizygote_count%0A%20%20%20%20%20%20%09%09%7D%0A%20%20%20%20%20%20%20%20%20%20af%0A%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20exome%20%7B%0A%20%20%20%20%20%20%20%20%20%20populations%20%7B%0A%20%20%20%20%20%20%09%09%20%20id%0A%20%20%20%20%20%20%20%20%20%20%20%20ac%0A%20%20%20%20%20%20%20%20%20%20%20%20an%0A%20%20%20%20%20%20%20%20%20%20%20%20homozygote_count%0A%20%20%20%20%20%20%20%20%20%20%20%20hemizygote_count%0A%20%20%20%20%20%20%09%09%7D%0A%20%20%20%20%20%20%20%20%20%20af%20%20%20%20%20%20%0A%09%09%09%09%7D%09%09%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D"


responsea = requests.get(gnomadurl + gnomadvar)
gnomadvarsa = responsea.json()
gnovarxx = gnomadvarsa.get('data')
gnovaryy = gnovarxx.get('variant')
gnovarzz = gnovaryy.get('exome')
gnovarpop = gnovarzz.get('populations')
gnomadaf = json_normalize(gnovarpop, max_level=0)
gnomadaf = gnomadaf.convert_dtypes()

gnomadaf[main + "del"] = gnomadaf['ac'] / gnomadaf['an']

gnomadafall = gnomadaf.drop(['ac', 
                      'an', 
                      'homozygote_count', 
                      'hemizygote_count'], axis=1)

retry_statuses = [502, 503, 504]

log_retryable = "failed_variants_retryable.log"
log_nonretryable = "failed_variants_nonretryable.log"


SNPs = pvars['variant'].unique().tolist()
#del SNPs[15500:30985]
#del SNPs[99:38555]

for variant in SNPs:
### Search by variant for allele frequencies
    
    gnomadurl = "https://gnomad.broadinstitute.org/api?"
    gnomadvar= "query=%20%20%20%20%7B%0A%09%09variant(variantId%3A%20%22" + variant + "%22%2C%20dataset%3A%20gnomad_r4)%20%7B%0A%20%20%20%20%20%20%20%20variant_id%0A%20%20%20%20%20%20%20%20transcript_consequences%20%7B%0A%20%20%20%20%20%20%20%20%20%20gene_symbol%0A%20%20%20%20%20%20%20%20%09gene_id%0A%20%20%20%20%20%20%20%20%20%20is_canonical%0A%20%20%20%20%20%20%20%20%20%20is_mane_select%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%09rsid%0A%20%20%20%20%20%20%09in_silico_predictors%20%7B%0A%20%20%20%20%20%20%09%20%20id%0A%20%20%20%20%20%20%20%20%20%20value%0A%20%20%20%20%20%20%09%7D%0A%20%20%20%20%20%20%20%20genome%20%7B%0A%20%20%20%20%20%20%09%09populations%20%7B%0A%20%20%20%20%20%20%09%09%20%20id%0A%20%20%20%20%20%20%20%20%20%20%20%20ac%0A%20%20%20%20%20%20%20%20%20%20%20%20an%0A%20%20%20%20%20%20%20%20%20%20%20%20homozygote_count%0A%20%20%20%20%20%20%20%20%20%20%20%20hemizygote_count%0A%20%20%20%20%20%20%09%09%7D%0A%20%20%20%20%20%20%20%20%20%20af%0A%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20exome%20%7B%0A%20%20%20%20%20%20%20%20%20%20populations%20%7B%0A%20%20%20%20%20%20%09%09%20%20id%0A%20%20%20%20%20%20%20%20%20%20%20%20ac%0A%20%20%20%20%20%20%20%20%20%20%20%20an%0A%20%20%20%20%20%20%20%20%20%20%20%20homozygote_count%0A%20%20%20%20%20%20%20%20%20%20%20%20hemizygote_count%0A%20%20%20%20%20%20%09%09%7D%0A%20%20%20%20%20%20%20%20%20%20af%20%20%20%20%20%20%0A%09%09%09%09%7D%09%09%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D"

    max_retries = 5
    delay = 2


    for attempt in range(max_retries):
        try:
            response = requests.get(gnomadurl + gnomadvar)

            if response.status_code in retry_statuses:
                raise requests.exceptions.HTTPError(f"Retryable error: {response.status_code}")

            response.raise_for_status()

            gnomadvars = response.json()
            gnovarx = gnomadvars.get('data')
            gnovary = gnovarx.get('variant')
            gnovarz = gnovary.get('genome')
            gnovarpop = gnovarz.get('populations')

            gnomadaf = json_normalize(gnovarpop, max_level=0)
            gnomadaf = gnomadaf.convert_dtypes()
            gnomadaf[variant] = gnomadaf['ac'] / gnomadaf['an']
            gnomadaf = gnomadaf.drop(['ac', 'an', 'homozygote_count', 'hemizygote_count'], axis=1)

            gnomadafall = pd.merge(gnomadafall, gnomadaf, on='id', how='outer')
            time.sleep(2)
            break

        except requests.exceptions.HTTPError as e:
            if response.status_code in retry_statuses:
                print(f"Retryable error ({response.status_code}) on attempt {attempt+1} for {variant}")
                time.sleep(delay)
                delay *= 2            
                if attempt == max_retries - 1:
                    with open(log_retryable, "a") as f:
                        f.write(variant + "\n")
                    print(f"Logged retryable failure for {variant}")
            else:
                print(f"Non-retryable HTTP error for {variant}: {e}")
                with open(log_nonretryable, "a") as f:
                    f.write(variant + "\n")
                time.sleep(2)
                break
        except Exception as e:
#            print(f"Unexpected error for {variant}: {e}")
            with open(log_nonretryable, "a") as f:
                f.write(variant + "\n")
            time.sleep(2)
            break


            
gnomadafall = gnomadafall.drop([main +"del"], axis=1)

gnomadafally = gnomadafall.T
gnomadafally.columns = gnomadafall.index
gnomadafally.index.name = 'SNP_ID'
gnomadafally = gnomadafally.reset_index()

gnomadafally.columns = gnomadafally.iloc[0]
gnomadafally = gnomadafally[1:]


gnomadafally.to_csv("Pathogenic_Variant_Frequencies_All.csv", index=False)


# In[4]:


pheno = pd.read_csv("Pathogenic_Variants.csv", delimiter=",")
pheno = pheno.drop(['SomaticClinicalImpact',
                   'SomaticClinicalImpactLastEvaluated',
                   'ReviewStatusClinicalImpact'], axis=1)

freq = pd.read_csv("Pathogenic_Variant_Frequencies_All.csv", delimiter=",")
freq = freq.drop(['hgdp:japanese',
                'hgdp:japanese_XX',
                'hgdp:japanese_XY',
                'hgdp:adygei',
                'hgdp:adygei_XX',
                'hgdp:adygei_XY',
                'hgdp:orcadian',
                'hgdp:orcadian_XX',
                'hgdp:orcadian_XY',
                'hgdp:bantusouthafrica',
                'hgdp:bantusouthafrica_XX',
                'hgdp:bantusouthafrica_XY',
                'hgdp:yakut',
                'hgdp:yakut_XX',
                'hgdp:yakut_XY',
                'hgdp:han',
                'hgdp:han_XX',
                'hgdp:han_XY',
                'hgdp:uygur',
                'hgdp:uygur_XX',
                'hgdp:uygur_XY',
                'hgdp:balochi',
                'hgdp:balochi_XX',
                'hgdp:balochi_XY',
                'hgdp:bedouin',
                'hgdp:bedouin_XX',
                'hgdp:bedouin_XY',
                'hgdp:russian',
                'hgdp:russian_XX',
                'hgdp:russian_XY',
                'hgdp:daur',
                'hgdp:daur_XX',
                'hgdp:daur_XY',
                'hgdp:pima',
                'hgdp:pima_XX',
                'hgdp:pima_XY',
                'hgdp:hezhen',
                'hgdp:hezhen_XX',
                'hgdp:hezhen_XY',
                'hgdp:biaka',
                'hgdp:biaka_XX',
                'hgdp:biaka_XY',
                'hgdp:miao',
                'hgdp:miao_XX',
                'hgdp:miao_XY',
                'hgdp:sindhi',
                'hgdp:sindhi_XX',
                'hgdp:sindhi_XY',
                'hgdp:northernhan',
                'hgdp:northernhan_XX',
                'hgdp:northernhan_XY',
                'hgdp:oroqen',
                'hgdp:oroqen_XX',
                'hgdp:oroqen_XY',
                'hgdp:san',
                'hgdp:san_XX',
                'hgdp:san_XY',
                'hgdp:tu',
                'hgdp:tu_XX',
                'hgdp:tu_XY',
                'hgdp:tuscan',
                'hgdp:tuscan_XX',
                'hgdp:tuscan_XY',
                'hgdp:mbuti',
                'hgdp:mbuti_XX',
                'hgdp:mbuti_XY',
                'hgdp:palestinian',
                'hgdp:palestinian_XX',
                'hgdp:palestinian_XY',
                'hgdp:tujia',
                'hgdp:tujia_XX',
                'hgdp:tujia_XY',
                'hgdp:druze',
                'hgdp:druze_XX',
                'hgdp:druze_XY',
                'hgdp:pathan',
                'hgdp:pathan_XX',
                'hgdp:pathan_XY',
                'hgdp:basque',
                'hgdp:basque_XX',
                'hgdp:basque_XY',
                'hgdp:makrani',
                'hgdp:makrani_XX',
                'hgdp:makrani_XY',
                'hgdp:burusho',
                'hgdp:burusho_XX',
                'hgdp:burusho_XY',
                'hgdp:mongolian',
                'hgdp:mongolian_XX',
                'hgdp:mongolian_XY',
                'hgdp:bougainville',
                'hgdp:bougainville_XX',
                'hgdp:bougainville_XY',
                'hgdp:papuansepik',
                'hgdp:papuansepik_XX',
                'hgdp:papuansepik_XY',
                'hgdp:yi',
                'hgdp:yi_XX',
                'hgdp:yi_XY',
                'hgdp:naxi',
                'hgdp:naxi_XX',
                'hgdp:naxi_XY',
                'hgdp:lahu',
                'hgdp:lahu_XX',
                'hgdp:lahu_XY',
                'hgdp:sardinian',
                'hgdp:sardinian_XX',
                'hgdp:sardinian_XY',
                'hgdp:karitiana',
                'hgdp:karitiana_XX',
                'hgdp:karitiana_XY',
                'hgdp:mozabite',
                'hgdp:mozabite_XX',
                'hgdp:mozabite_XY',
                'hgdp:yoruba',
                'hgdp:yoruba_XX',
                'hgdp:yoruba_XY',
                'hgdp:dai',
                'hgdp:dai_XX',
                'hgdp:dai_XY',
                'hgdp:bergamoitalian',
                'hgdp:bergamoitalian_XX',
                'hgdp:bergamoitalian_XY',
                'hgdp:cambodian',
                'hgdp:cambodian_XX',
                'hgdp:cambodian_XY',
                'hgdp:french',
                'hgdp:french_XX',
                'hgdp:french_XY',
                'hgdp:mandenka',
                'hgdp:mandenka_XX',
                'hgdp:mandenka_XY',
                'hgdp:surui',
                'hgdp:surui_XX',
                'hgdp:surui_XY',
                'hgdp:brahui',
                'hgdp:brahui_XX',
                'hgdp:brahui_XY',
                'hgdp:hazara',
                'hgdp:hazara_XX',
                'hgdp:hazara_XY',
                'hgdp:kalash',
                'hgdp:kalash_XX',
                'hgdp:kalash_XY',
                'hgdp:papuanhighlands',
                'hgdp:papuanhighlands_XX',
                'hgdp:papuanhighlands_XY',
                'hgdp:xibo',
                'hgdp:xibo_XX',
                'hgdp:xibo_XY',
                'hgdp:colombian',
                'hgdp:colombian_XX',
                'hgdp:colombian_XY',
                'hgdp:bantukenya',
                'hgdp:bantukenya_XX',
                'hgdp:bantukenya_XY',
                'hgdp:she',
                'hgdp:she_XX',
                'hgdp:she_XY',
                'hgdp:maya',
                'hgdp:maya_XX',
                'hgdp:maya_XY',
                'hgdp:XX',
                'hgdp:XY',
                '1kg:esn',
                '1kg:esn_XX',
                '1kg:esn_XY',
                '1kg:pur',
                '1kg:pur_XX',
                '1kg:pur_XY',
                '1kg:pjl',
                '1kg:pjl_XX',
                '1kg:pjl_XY',
                '1kg:clm',
                '1kg:clm_XX',
                '1kg:clm_XY',
                '1kg:jpt',
                '1kg:jpt_XX',
                '1kg:jpt_XY',
                '1kg:chb',
                '1kg:chb_XX',
                '1kg:chb_XY',
                '1kg:stu',
                '1kg:stu_XX',
                '1kg:stu_XY',
                '1kg:itu',
                '1kg:itu_XX',
                '1kg:itu_XY',
                '1kg:tsi',
                '1kg:tsi_XX',
                '1kg:tsi_XY',
                '1kg:mxl',
                '1kg:mxl_XX',
                '1kg:mxl_XY',
                '1kg:ceu',
                '1kg:ceu_XX',
                '1kg:ceu_XY',
                '1kg:msl',
                '1kg:msl_XX',
                '1kg:msl_XY',
                '1kg:beb',
                '1kg:beb_XX',
                '1kg:beb_XY',
                '1kg:yri',
                '1kg:yri_XX',
                '1kg:yri_XY',
                '1kg:fin',
                '1kg:fin_XX',
                '1kg:fin_XY',
                '1kg:khv',
                '1kg:khv_XX',
                '1kg:khv_XY',
                '1kg:cdx',
                '1kg:cdx_XX',
                '1kg:cdx_XY',
                '1kg:lwk',
                '1kg:lwk_XX',
                '1kg:lwk_XY',
                '1kg:acb',
                '1kg:acb_XX',
                '1kg:acb_XY',
                '1kg:asw',
                '1kg:asw_XX',
                '1kg:asw_XY',
                '1kg:ibs',
                '1kg:ibs_XX',
                '1kg:ibs_XY',
                '1kg:gbr',
                '1kg:gbr_XX',
                '1kg:gbr_XY',
                '1kg:pel',
                '1kg:pel_XX',
                '1kg:pel_XY',
                '1kg:gih',
                '1kg:gih_XX',
                '1kg:gih_XY',
                '1kg:chs',
                '1kg:chs_XX',
                '1kg:chs_XY',
                '1kg:gwd',
                '1kg:gwd_XX',
                '1kg:gwd_XY',
                '1kg:XX',
                '1kg:XY'], axis=1)



joined = pd.merge(pheno, freq, left_on='variant', right_on='id', how='inner', validate='m:m')
joined = joined.convert_dtypes()

columns_to_sum = joined.columns[16:48]


joined = joined.drop_duplicates(subset=['variant', 'PhenotypeList'])
# Step 3: Group by 'gene' and 'phenotypes' and sum
grouped = joined.groupby(['GeneSymbol', 'PhenotypeList'])[columns_to_sum].sum().reset_index()

grouped = grouped.merge(joined[['GeneSymbol', 'Chromosome']], left_on='GeneSymbol', right_on='GeneSymbol', how='left')
grouped = grouped.drop_duplicates(subset=['GeneSymbol', 'PhenotypeList'])
# Optional: Save the result
grouped.to_csv("Summmed_Frequencies.csv", index=False)


# In[7]:


## Convert Frequences to Prevalences
#### Using Hardy-Weinberg Equilibrium ... A^2 + 2AB + B^2 = 1
###### For autosomes
#### Carrier frequency = 2AB + B^2
#### Homozygote frequency = B^2
###### For genes on the X chromosome
#### Carrier frequency = (2AB + B^2)/2 + (B/2)
#### Homozygote frequency = (B^2)/2 + (B/2)

popn = pd.read_csv("popn.txt", delimiter=",")
popnx = popn.set_index('ABV').T.to_dict('list')

for key, values in popnx.items():
    abv = key
    full = values[0]
    
    def calculate_carrier_prevalence(row):
        if pd.isna(row[abv + "_XY"]) or pd.isna(row['Chromosome']):
            return pd.NA
        elif row['Chromosome'] == 'X':
            return row[abv + "_XY"] * 100000
        else:
            return ((2 * (row[abv + "_XY"] * (1 - row[abv + "_XY"]))) + (row[abv + "_XY"]**2)) * 100000
    
    grouped[full + ' Male Carrier Prevalence'] = grouped.apply(calculate_carrier_prevalence, axis=1)
    
for key, values in popnx.items():
    abv = key
    full = values[0]
    
    grouped[full + ' Female Carrier Prevalence'] = ((2 * (grouped[abv + "_XX"] * (1 - grouped[abv + "_XX"]))) + (grouped[abv + "_XX"]**2)) * 100000
    
    
for key, values in popnx.items():
    abv = key
    full = values[0]
    
    def calculate_all_carrier_prevalence(row):
        if pd.isna(row[abv]) or pd.isna(row['Chromosome']):
            return pd.NA
        elif row['Chromosome'] == 'X':
            return ((((2 * (row[abv] * (1 - row[abv]))) + (row[abv]**2))/2) + (row[abv]/2)) * 100000
        else:
            return ((2 * (row[abv] * (1 - row[abv]))) + (row[abv]**2)) * 100000
    
    grouped[full + ' Carrier Prevalence'] = grouped.apply(calculate_all_carrier_prevalence, axis=1)
    
    
for key, values in popnx.items():
    abv = key
    full = values[0]
    
    def calculate_hom_carrier_prevalence(row):
        if pd.isna(row[abv + "_XY"]) or pd.isna(row['Chromosome']):
            return pd.NA
        elif row['Chromosome'] == 'X':
            return row[abv + "_XY"] * 100000
        else:
            return (row[abv + "_XY"]**2) * 100000
    
    grouped[full + ' Male Hom Carrier Prevalence'] = grouped.apply(calculate_hom_carrier_prevalence, axis=1)
    
for key, values in popnx.items():
    abv = key
    full = values[0]
    
    grouped[full + ' Female Hom Carrier Prevalence'] = (grouped[abv + "_XX"]**2) * 100000
    
    
for key, values in popnx.items():
    abv = key
    full = values[0]
    
    def calculate_all_hom_carrier_prevalence(row):
        if pd.isna(row[abv]) or pd.isna(row['Chromosome']):
            return pd.NA
        elif row['Chromosome'] == 'X':
            return (((row[abv]**2)/2) + (row[abv]/2)) * 100000
        else:
            return (row[abv]**2) * 100000
    
    grouped[full + ' Hom Carrier Prevalence'] = grouped.apply(calculate_all_hom_carrier_prevalence, axis=1)


    
grouped = grouped.drop(['remaining',
                'remaining_XX',
                'remaining_XY',
                'amr',
                'amr_XX',
                'amr_XY',
                'fin',
                'fin_XX',
                'fin_XY',
                'mid',
                'mid_XX',
                'mid_XY',
                'nfe',
                'nfe_XX',
                'nfe_XY',
                'sas',
                'sas_XX',
                'sas_XY',
                'asj',
                'asj_XX',
                'asj_XY',
                'eas',
                'eas_XX',
                'eas_XY',
                'afr',
                'afr_XX',
                'afr_XY',
                'ami',
                'ami_XX',
                'ami_XY',
                'XX',
                'XY'], axis=1)

grouped = grouped.convert_dtypes()   
grouped.to_csv("Gene_Based_Prevalences.csv", index=False)


# In[8]:


#Carrier Frequency

## Import Results for Graphical Representation

carprevtot = pd.read_csv("Gene_Based_Prevalences.csv", delimiter=",")
carprevtot = carprevtot.convert_dtypes()

## Tidy Columns

carprevtot = carprevtot.drop(['Chromosome',
' European (excluding Finland) Male Carrier Prevalence',
' Finnish Male Carrier Prevalence',
' American (admixed) Male Carrier Prevalence',
' African/African American Male Carrier Prevalence',
' Amish Male Carrier Prevalence',
' Middle Eastern Male Carrier Prevalence',
' South Asian Male Carrier Prevalence',
' East Asian Male Carrier Prevalence',
' Ashkenazi Jewish Male Carrier Prevalence',
' European (excluding Finland) Female Carrier Prevalence',
' Finnish Female Carrier Prevalence',
' American (admixed) Female Carrier Prevalence',
' African/African American Female Carrier Prevalence',
' Amish Female Carrier Prevalence',
' Middle Eastern Female Carrier Prevalence',
' South Asian Female Carrier Prevalence',
' East Asian Female Carrier Prevalence',
' Ashkenazi Jewish Female Carrier Prevalence',
' European (excluding Finland) Male Hom Carrier Prevalence',
' Finnish Male Hom Carrier Prevalence',
' American (admixed) Male Hom Carrier Prevalence',
' African/African American Male Hom Carrier Prevalence',
' Amish Male Hom Carrier Prevalence',
' Middle Eastern Male Hom Carrier Prevalence',
' South Asian Male Hom Carrier Prevalence',
' East Asian Male Hom Carrier Prevalence',
' Ashkenazi Jewish Male Hom Carrier Prevalence',
' European (excluding Finland) Female Hom Carrier Prevalence',
' Finnish Female Hom Carrier Prevalence',
' American (admixed) Female Hom Carrier Prevalence',
' African/African American Female Hom Carrier Prevalence',
' Amish Female Hom Carrier Prevalence',
' Middle Eastern Female Hom Carrier Prevalence',
' South Asian Female Hom Carrier Prevalence',
' East Asian Female Hom Carrier Prevalence',
' Ashkenazi Jewish Female Hom Carrier Prevalence',
' European (excluding Finland) Hom Carrier Prevalence',
' Finnish Hom Carrier Prevalence',
' American (admixed) Hom Carrier Prevalence',
' African/African American Hom Carrier Prevalence',
' Amish Hom Carrier Prevalence',
' Middle Eastern Hom Carrier Prevalence',
' South Asian Hom Carrier Prevalence',
' East Asian Hom Carrier Prevalence',
' Ashkenazi Jewish Hom Carrier Prevalence'], axis=1)


carprevtot = carprevtot.rename(columns={' European (excluding Finland) Carrier Prevalence': 'European \n (excl Finland)', ' Finnish Carrier Prevalence': 'Finnish', ' American (admixed) Carrier Prevalence': 'American \n (admixed)', ' African/African American Carrier Prevalence': 'African/ \n African American', ' Amish Carrier Prevalence': 'Amish', ' Middle Eastern Carrier Prevalence': 'Middle Eastern', ' South Asian Carrier Prevalence': 'South Asian', ' East Asian Carrier Prevalence': 'East Asian', ' Ashkenazi Jewish Carrier Prevalence': 'Ashkenazi Jewish',})


prevalence_columns = ['European \n (excl Finland)', 'Finnish', 'American \n (admixed)', 'African/ \n African American', 'Amish', 'Middle Eastern', 'South Asian', 'East Asian', 'Ashkenazi Jewish']


max_value = carprevtot[prevalence_columns].max().max()
# Round up to the nearest 100
vmax = int(np.ceil(max_value / 100.0)) * 100

# Get unique phenotypes
phenotypes = carprevtot['PhenotypeList'].unique()


# Create a multi-page PDF
with PdfPages('carrier_prevalence_heatmaps_by_phenotype.pdf') as pdf:
    for phenotype in phenotypes:
        subset = carprevtot[carprevtot['PhenotypeList'] == phenotype].copy()
        subset['Label'] = subset['GeneSymbol']
        subset.set_index('Label', inplace=True)
        heatmap_data = subset[prevalence_columns]

#Set figure height dynamically
        row_height = 1.5
        fig_height = max(4.5, len(heatmap_data) * row_height)

        fig, ax = plt.subplots(figsize=(10, fig_height))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='inferno_r',
            fmt=".1f",
            linewidths=0.5,
            vmin=0,
            vmax=vmax,
            ax=ax
        )
        ax.set_title(f'Carrier Prevalence by Population for Genes Causing \n {phenotype}')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_tick_params(rotation=50)
        ax.set_xlabel('Population')
        ax.set_ylabel('Gene')
        ax.yaxis.set_tick_params(rotation=0)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)




# Get unique genes
genesg = carprevtot['GeneSymbol'].unique()


# Create a multi-page PDF
with PdfPages('carrier_prevalence_heatmaps_by_gene.pdf') as pdf:
    for genes in genesg:
        subset = carprevtot[carprevtot['GeneSymbol'] == genes].copy()
        subset['Label'] = subset['PhenotypeList']
        subset.set_index('Label', inplace=True)
        heatmap_data = subset[prevalence_columns]

#Set figure height dynamically
        row_height = 1.5  
        fig_width = 10
        fig_height = max(4.5, len(heatmap_data) * row_height)

        fig, ax = plt.subplots(figsize=(10, fig_height))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='inferno_r',
            fmt=".1f",
            linewidths=0.5,
            vmin=0,
            vmax=vmax,
            ax=ax
        )
        ax.set_title(f'Carrier Prevalence by Population for Phenotypes Caused by \n {genes}')
        ax.xaxis.set_ticks_position('top')
        ax.set_xlabel('Population')
        ax.xaxis.set_tick_params(rotation=50)
        ax.set_ylabel('Phenotype')
        wrapped_labels = [textwrap.fill(label, width=20, break_long_words=False) for label in heatmap_data.index]
        ax.set_yticklabels(wrapped_labels, rotation=0)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)


# In[9]:


#Homozygote Carrier Frequency

## Import Results for Graphical Representation

homprevtot = pd.read_csv("Gene_Based_Prevalences.csv", delimiter=",")
homprevtot = homprevtot.convert_dtypes()

## Tidy Columns

homprevtot = homprevtot.drop(['Chromosome',
' European (excluding Finland) Male Carrier Prevalence',
' Finnish Male Carrier Prevalence',
' American (admixed) Male Carrier Prevalence',
' African/African American Male Carrier Prevalence',
' Amish Male Carrier Prevalence',
' Middle Eastern Male Carrier Prevalence',
' South Asian Male Carrier Prevalence',
' East Asian Male Carrier Prevalence',
' Ashkenazi Jewish Male Carrier Prevalence',
' European (excluding Finland) Female Carrier Prevalence',
' Finnish Female Carrier Prevalence',
' American (admixed) Female Carrier Prevalence',
' African/African American Female Carrier Prevalence',
' Amish Female Carrier Prevalence',
' Middle Eastern Female Carrier Prevalence',
' South Asian Female Carrier Prevalence',
' East Asian Female Carrier Prevalence',
' Ashkenazi Jewish Female Carrier Prevalence',
' European (excluding Finland) Carrier Prevalence',
' Finnish Carrier Prevalence',
' American (admixed) Carrier Prevalence',
' African/African American Carrier Prevalence',
' Amish Carrier Prevalence',
' Middle Eastern Carrier Prevalence',
' South Asian Carrier Prevalence',
' East Asian Carrier Prevalence',
' Ashkenazi Jewish Carrier Prevalence',
' European (excluding Finland) Male Hom Carrier Prevalence',
' Finnish Male Hom Carrier Prevalence',
' American (admixed) Male Hom Carrier Prevalence',
' African/African American Male Hom Carrier Prevalence',
' Amish Male Hom Carrier Prevalence',
' Middle Eastern Male Hom Carrier Prevalence',
' South Asian Male Hom Carrier Prevalence',
' East Asian Male Hom Carrier Prevalence',
' Ashkenazi Jewish Male Hom Carrier Prevalence',
' European (excluding Finland) Female Hom Carrier Prevalence',
' Finnish Female Hom Carrier Prevalence',
' American (admixed) Female Hom Carrier Prevalence',
' African/African American Female Hom Carrier Prevalence',
' Amish Female Hom Carrier Prevalence',
' Middle Eastern Female Hom Carrier Prevalence',
' South Asian Female Hom Carrier Prevalence',
' East Asian Female Hom Carrier Prevalence',
' Ashkenazi Jewish Female Hom Carrier Prevalence'], axis=1)


homprevtot = homprevtot.rename(columns={' European (excluding Finland) Hom Carrier Prevalence': 'European \n (excl Finland)', ' Finnish Hom Carrier Prevalence': 'Finnish', ' American (admixed) Hom Carrier Prevalence': 'American \n (admixed)', ' African/African American Hom Carrier Prevalence': 'African/ \n African American', ' Amish Hom Carrier Prevalence': 'Amish', ' Middle Eastern Hom Carrier Prevalence': 'Middle Eastern', ' South Asian Hom Carrier Prevalence': 'South Asian', ' East Asian Hom Carrier Prevalence': 'East Asian', ' Ashkenazi Jewish Hom Carrier Prevalence': 'Ashkenazi Jewish',})


prevalence_columns = ['European \n (excl Finland)', 'Finnish', 'American \n (admixed)', 'African/ \n African American', 'Amish', 'Middle Eastern', 'South Asian', 'East Asian', 'Ashkenazi Jewish']


max_value = homprevtot[prevalence_columns].max().max()
# Round up to the nearest 100
vmax = int(np.ceil(max_value / 10000.0)) * 10000

# Get unique phenotypes
phenotypes = homprevtot['PhenotypeList'].unique()


# Create a multi-page PDF
with PdfPages('homozygote_carrier_prevalence_heatmaps_by_phenotype.pdf') as pdf:
    for phenotype in phenotypes:
        subset = homprevtot[homprevtot['PhenotypeList'] == phenotype].copy()
        subset['Label'] = subset['GeneSymbol']
        subset.set_index('Label', inplace=True)
        heatmap_data = subset[prevalence_columns]

#Set figure height dynamically
        row_height = 1.5
        fig_height = max(4.5, len(heatmap_data) * row_height)

        fig, ax = plt.subplots(figsize=(10, fig_height))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='inferno_r',
            fmt=".1f",
            linewidths=0.5,
            vmin=0,
            vmax=vmax,
            ax=ax
        )
        ax.set_title(f'"Homozygote" Carrier Prevalence by Population for Genes Causing \n {phenotype}')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_tick_params(rotation=50)
        ax.set_xlabel('Population')
        ax.set_ylabel('Gene')
        ax.yaxis.set_tick_params(rotation=0)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)




# Get unique genes
genesg = homprevtot['GeneSymbol'].unique()


# Create a multi-page PDF
with PdfPages('homozygote_carrier_prevalence_heatmaps_by_gene.pdf') as pdf:
    for genes in genesg:
        subset = homprevtot[homprevtot['GeneSymbol'] == genes].copy()
        subset['Label'] = subset['PhenotypeList']
        subset.set_index('Label', inplace=True)
        heatmap_data = subset[prevalence_columns]

#Set figure height dynamically
        row_height = 1.5  
        fig_width = 10
        fig_height = max(4.5, len(heatmap_data) * row_height)

        fig, ax = plt.subplots(figsize=(10, fig_height))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='inferno_r',
            fmt=".1f",
            linewidths=0.5,
            vmin=0,
            vmax=vmax,
            ax=ax
        )
        ax.set_title(f'"Homozygote" Carrier Prevalence by Population for Phenotypes Caused by \n {genes}')
        ax.xaxis.set_ticks_position('top')
        ax.set_xlabel('Population')
        ax.xaxis.set_tick_params(rotation=50)
        ax.set_ylabel('Phenotype')
        wrapped_labels = [textwrap.fill(label, width=20, break_long_words=False) for label in heatmap_data.index]
        ax.set_yticklabels(wrapped_labels, rotation=0)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)
