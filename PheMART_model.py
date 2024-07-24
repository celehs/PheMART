#!/usr/bin/env python
# coding: utf-8
# In[1]:
from sklearn.metrics import auc, precision_score, recall_score, accuracy_score, confusion_matrix, f1_score, \
    roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
import pandas as pd
import csv
import random
import logging, os
from tensorflow.keras import layers, models
import tensorflow as tf
from sklearn import metrics
from a_utilization_beforeAE_joint_ppi_interface_VC import loaddata
import scipy.stats as ss
import time
import argparse
tf.keras.backend.set_floatx('float32')

# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--flag_reload', type=int, default=0,
                        help='0-1 to denote if reload the model')
    parser.add_argument('--flag_modelsave', type=int, default=0,
                        help='0-1 to denote if save the model')
    parser.add_argument('--epochs', type=int, default=60,
                        help='epochs')
    parser.add_argument('--flag_debug', type=int, default=0,
                        help='flag to indicate debug')
    parser.add_argument('--flag_negative_filter', type=int, default=1,
                        help='flag for negative_filter')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='train_ratio')
    parser.add_argument('--latent_dim', type=int, default=80,
                        help='embedding dim')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.0006,
                        help='learning_rate')
    parser.add_argument('--weight_cosine', type=float, default=5.0,
                        help='weight_cosine')
    parser.add_argument('--weight_vae', type=float, default=0.3,
                        help='weight_vae')
    parser.add_argument('--weight_distill', type=float, default=0.1,
                        help='weight_distill')
    parser.add_argument('--weight_unlabel_snps', type=float, default=0.2,
                        help='weight_unlabel_snps')
    parser.add_argument('--weight_CLIP', type=float, default=0.8,
                        help='weight_CLIP')
    parser.add_argument('--weight_CLIP_snps', type=float, default=0.5,
                        help='weight_CLIP_snps')
    parser.add_argument('--weight_CLIP_cui', type=float, default=0.5,
                        help='weight_CLIP_cui')
    parser.add_argument('--negative_disease', type=int, default=100,
                        help='number of negative_disease')
    parser.add_argument('--negative_snps', type=int, default=100,
                        help='number of sampled negative_snps')
    parser.add_argument('--margin_same', type=float, default=0.1,
                        help='margin_same')
    parser.add_argument('--margin_ppi', type=float, default=0.0,
                        help='margin_ppi')
    parser.add_argument('--margin_differ', type=float, default=-0.2,
                        help='margin_differ')
    parser.add_argument('--tau_softmax', type=float, default=0.1,
                        help='tau_softmax for CLIP')
    parser.add_argument('--flag_hard_negative', type=int, default=1,
                        help='if using flag_hard_negative')
    parser.add_argument('--model_savename', type=str,default="interaction_dim72_negative_mining",
                        help='model name to save ')
    parser.add_argument('--flag_save_unlabel_emb', type=int, default=0,
                        help='if predict and save unlabeled snps embeddings')
    parser.add_argument('--flag_cross_cui', type=int, default=0,
                        help='if flag_cross_cui validation')
    parser.add_argument('--flag_cross_gene', type=int, default=0,
                        help='if cross-gene validation')
    parser.add_argument('--content_unlabel', type=str, default="UDN",
                        help='what unlabel SNPs to predict')
    parser.add_argument('--flag_save_unlabel_predict', type=int, default=1,
                        help='if predict and save unlabeled snps predictions')
    parser.add_argument('--savename_unlabel_predict', type=str, default="snps_prediction_unlabeled_all.csv",
                        help='filename to save unlabeled snps predictions')
    parser.add_argument('--dirr_results_main', type=str, default="results/",
                        help='Directory of to save results ')
    parser.add_argument('--dirr_save_model', type=str, default="Model_save/",
                        help='Directory of to save models ')
    parser.add_argument('--dirr_results', type=str, default="/trait_dim72/",
                        help='Directory to save results including embedings of train/test/unlabelel and unlabled predictions')
    parser.add_argument('--filename_eval', type=str,default="result_trait_dim72.txt",
                        help='filename to save the prediction evaluations')
    args = parser.parse_args()
    return args

PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ARGS = parse_arguments(PARSER)
flag_reload=ARGS.flag_reload  #False
flag_modelsave=ARGS.flag_modelsave  #False
epochs=ARGS.epochs   #   55
latent_dim=ARGS.latent_dim   # 72
batch_size=ARGS.batch_size   #  512
learning_rate=ARGS.learning_rate   #  0.0006
weight_cosine=ARGS.weight_cosine   #  1.0
weight_vae= ARGS.weight_vae   #  0.3
weight_unlabel_snps=ARGS.weight_unlabel_snps   #   0.2
weight_CLIP=ARGS.weight_CLIP   #  0.5
train_ratio=ARGS.train_ratio   #   0.9
content_unlabel=ARGS.content_unlabel
flag_negative_filter=ARGS.flag_negative_filter
flag_cross_gene=ARGS.flag_cross_gene
flag_cross_cui=ARGS.flag_cross_cui
negative_disease=ARGS.negative_disease   #   100
negative_snps=ARGS.negative_snps   #  100
margin_same=ARGS.margin_same   #  0.8
margin_differ=ARGS.margin_differ   #   -0.2
weight_CLIP_snps=ARGS.weight_CLIP_snps   #     0.25
weight_CLIP_cui= ARGS.weight_CLIP_cui   #   0.75
tau_softmax=ARGS.tau_softmax   #   0.1
flag_hard_negative=ARGS.flag_hard_negative   #  1
flag_debug=ARGS.flag_debug
model_savename=ARGS.model_savename   # "Interaction_CODER_semi_align"
flag_save_unlabel_emb=ARGS.flag_save_unlabel_emb   # "Interaction_CODER_semi_align"
flag_save_unlabel_predict=ARGS.flag_save_unlabel_predict   # "Interaction_CODER_semi_align"
savename_unlabel_predict=ARGS.savename_unlabel_predict   # "Interaction_CODER_semi_align"
dirr_save_model= ARGS.dirr_save_model #       "Model_save/"
dirr_results_main=ARGS.dirr_results_main
filename_eval=ARGS.filename_eval
weight_distill=ARGS.weight_distill
margin_ppi=ARGS.margin_ppi
scale_AE= 50.0   #   50.0
weight_kl=0.0  #   0.0
time_preprocessing=0
epoch_show = 3
epoch_MRR=500
if ARGS.epochs<3:
    learning_rate=0.0000001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

if not os.path.exists(dirr_results_main):
    os.mkdir(dirr_results_main)
dirr_results=ARGS.dirr_results
if not os.path.exists(dirr_results):
    os.mkdir(dirr_results)
dirr_results=ARGS.dirr_results+"run_"+str(random.randint(0,10000))+"/"
if not os.path.exists(dirr_results):
    os.mkdir(dirr_results)
dirr_results_prediction=dirr_results+"predictions_SNPs/"
if not os.path.exists(dirr_results_prediction):
    os.mkdir(dirr_results_prediction)

dirr="/n/data1/hsph/biostat/celehs/lab/junwen/SNPs_codes/datasets/"
file_CUIs_target="CUIs_all_covered_CODER_0410_4749_noOTHERS_save.csv"
file_CUIs_target_UDN="HPO_terms_validation_all_reporteded.csv"
label_file="Clivar_snps_disease_Pathogenic_Likely_4_11_0410_trait_with_ALL_no_thers_1024.csv"
file_label_all="clinvar_missense_labeled_4_11.csv"
snps_label_file="labeled_230201_clinvar.csv"
snps_label_file_miss="snps_miss_without_snp_embedding_filter.csv"
snps_label_file_embedding_miss="snps_miss_without_snp_embedding_filter_embedding_seq.npy"
file_coexpression="gene_coexpression_gene2vec_dim_200_iter_9.txt"
file_emb_wild_unipro="uniprotid_gene_seq_embed.csv"
snps_label_file_embedding="labeled_230201_clinvar_embedding.npy"
disease_file_embedding="embeddings_coder_clinvar_traits_cui_mapped_ALL_0410_with_all_renamed_withCUI_noOthers.csv" # node_disease_CUI_mapped_CUI_embedding.npy
disease_file_embedding_EHR="Epoch_1000_L1_L2_3layer_MGB_VA_emb_embeddings_coder_clinvar_traits_cui_mapped_ALL_0410_with_all_renamed_withCUI_noOthers.csv"
# node_disease_CUI_mapped_CUI_embedding.csv node_disease_CUI_mapped_sapbert.csv
file_emb_wild="wildtype_embeddings.csv"
file_snps_gene_map="Mapping_snps_genes.csv"
snps_unlabel_file="unlabeled_230201_clinvar.csv"
snps_unlabel_file_embedding="unlabeled_230201_clinvar_embedding.npy"
clinvar_missense="clinvar_missense_labeled_4_11.csv"
file_pathogenic_no_traits="Clinar_missense_pathogenic_no_traits.csv"
genes_omim=set(list(pd.read_csv(dirr+"omim_clinvar_diff.csv")["gene"]))
file_hpo_embedding="UDN_HPO_term_ALL_with_name_embedding_coder.csv"
file_snps_gwas="GWAS_missense_label_HGv_no_overlap.csv"
snps_gwas=list(pd.read_csv(dirr+file_snps_gwas)["HGV"])
print ("snps_gwas:  len: ",len(snps_gwas))
genes_OMIM=list(pd.read_csv(dirr+"omim_cui_mapping_coderpp_refine_sorted_subset.csv")["gene"])
print("-----------------------genes_OMIM len:  ",len(set(genes_OMIM)))
genes_OMIM_noClinVar=list(pd.read_csv(dirr+"omim_cui_mapping_coderpp_refine_sorted_qiao_no_clinvar.csv")["gene"])
print("-----------------------genes_OMIM_noClinVar len:  ",len(set(genes_OMIM_noClinVar)))
file_snps_UDN="UDN_SNP_disease_relation_split_embedding.csv"
df_UND=pd.read_csv(dirr+file_snps_UDN)
columns=list(df_UND.columns)
embedding_UDN_shipa=np.array(df_UND[columns[13:]])
print("---embedding_UDN: ",embedding_UDN_shipa.shape)
snps_UDN_shipa=list(df_UND["Name"])
gene_name_UDN_shipa=list(df_UND["gene_name"])
print ("------snps_UDN:  len: ",len(snps_UDN_shipa))
file_snps_list_UDN="variant_12observation_count_thould0.csv"
snps_UDN_additional=list(pd.read_csv(dirr+file_snps_list_UDN)["clinvar_name"])

df = pd.read_csv(dirr+file_label_all)
dic_snp_year={}
if "year_"in content_unlabel:
    year_target=str(content_unlabel).split("_")[1]
    for  snps,sig in zip(df["Name"],df["Clinical significance (Last reviewed)"]):
        snps=str(snps).strip()
        sig=str(sig).strip()
        if str(sig)[-5:-1]==year_target:
            dic_snp_year[snps]=1

dic_HPO_CUI_valid={}

dic_snps_cui={}
dic_cui_snps={}
dic_cui_emb={}
dic_snps_emb={}
dic_snpsname_emb={}
dic_snpsname_emb_un={}
dic_snps_index = {}
dic_index_snps={}
dic_snps_gene={}
dic_snpsindex_gene={}

df=pd.read_csv(dirr+clinvar_missense)
SNPs_clinvar_missense=list(df["Name"])
SNPs_clinvar_missense=list({}.fromkeys(SNPs_clinvar_missense).keys())
print ("--------------SNPs_clinvar_missense len: ", len(SNPs_clinvar_missense))
df=pd.read_csv(dirr+file_pathogenic_no_traits)
Clinar_missense_pathogenic_no_traits=set(list(df["snp"]))


dic_gene_corexpression={}
lines=open(dirr+file_coexpression).readlines()
print("lines",len(lines))
embedding_all_gene_expression=[]
for line in lines:
    line=line.strip()
    gene=line.split()[0]
    vec=np.array([float(x) for x in line.split()[1:]])
    dic_gene_corexpression[gene]=vec
    embedding_all_gene_expression.append(vec)
print("---------------------dic_gene_corexpression",len(dic_gene_corexpression))
genes_all_with_coexpression=list(dic_gene_corexpression.keys())

##################################gene embedding######
dic_gene_emb={}
df=pd.read_csv(dirr+file_emb_wild_unipro)
genes=list(df["gene"])
features_all=np.array(df[df.columns[2:]])
print ("genes len: ",len(genes))
print ("features_all shape: ",features_all.shape)

gene_embedding_all=[]
for rowi in range(len(genes)):
    gene = str(genes[rowi]).strip()

    dic_gene_emb[gene] = features_all[rowi, :]

df=pd.read_csv(dirr+file_emb_wild)
genes=list(df["gene"])
features_all=np.array(df[df.columns[1:]])
for rowi in range(len(genes)):
    gene = str(genes[rowi]).strip()
    dic_gene_emb[gene] = features_all[rowi, :]

##########################dic snps gene
df=pd.read_csv(dirr+file_snps_gene_map)
snps_save=df["snps"]
gene_save=df["genes"]
for snps,gene in zip(snps_save,gene_save):
    dic_snps_gene[snps]=gene
print ("dic_snps_gene len: ",len(dic_snps_gene))

##########################dic snps gene
dic_gene_labeled={}
df = pd.read_csv(dirr+label_file)
snps_labeled_all=list({}.fromkeys(list(df["snps"])).keys())
for index, snps in zip(df["snps_index"],df["snps"]):
    snps=str(snps).strip()
    if snps in dic_snps_gene:
        gene = dic_snps_gene[snps]
        dic_gene_labeled[gene] = 1
        dic_snps_index[str(snps).strip()] = int(index)
        dic_index_snps[int(index)]=str(snps).strip()
print("snps_labeled_all labeled with diseases: ",len(snps_labeled_all))
print("dic_gene_labeled len------all labeled genes : ",len(dic_gene_labeled),"-----------------")

df=pd.read_csv(dirr+"udn_hpo2cui_mapping_0410.csv")
hpos=df["hpo"]
cuis=df["cui"]
sims=df["sim"]
for hpo, cui, sim, in zip(hpos,cuis,sims):
    if float(sim)>0.8:
        dic_HPO_CUI_valid[hpo]=cui

CUIs_all_covered=list(pd.read_csv(dirr+file_CUIs_target)["CUIs"])
df=pd.read_csv(dirr+"CUI_HPO_UMLS2021AB.csv")
hpos=df["code"]
cuis=df["cui"]
for cui,hpo in zip(cuis,hpos):
    if cui in CUIs_all_covered:
        dic_HPO_CUI_valid[hpo]=cui

###########snps embedding #############################snps embedding ##################
embedding=np.load(dirr+snps_label_file_embedding)
embedding=np.array(embedding)
print ("SNPs embedding shape labeled: ",embedding.shape)
df = pd.read_csv(dirr+snps_label_file)
SNPs=list(df["Name"])
print (" labeled SNPs: ",len(SNPs))
dic_snpsname_emb_binary={}
dic_snpsname_emb_all={}
dic_snpsname_emb_sameGENE={}

for rowi in range(len(SNPs)):
    snps_i=str(SNPs[rowi]).strip()
    if  str(SNPs[rowi]).strip() in dic_snps_gene and dic_snps_gene[str(SNPs[rowi]).strip()] in dic_gene_emb:
        gene=dic_snps_gene[snps_i]
        dic_snpsname_emb_binary[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])
        dic_snpsname_emb_all[snps_i] = np.array(embedding[rowi, :])

        if gene in dic_gene_labeled:
            dic_snpsname_emb_sameGENE[snps_i]=1
        if snps_i in dic_snps_index :
            dic_snps_emb[dic_snps_index[str(SNPs[rowi]).strip()]]=np.array(embedding[rowi,:])
            dic_snpsname_emb[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])

######unlabeled
embedding=np.load(dirr+snps_unlabel_file_embedding)
embedding=np.array(embedding)
print ("SNPs embedding shape unlabeled: ",embedding.shape)
df = pd.read_csv(dirr+snps_unlabel_file)
SNPs=list(df["Name"])
print (" unlabeled SNPs: ",len(SNPs))
for rowi in range(len(SNPs)):
    snps_i = str(SNPs[rowi]).strip()
    if str(SNPs[rowi]).strip() in dic_snps_gene and dic_snps_gene[str(SNPs[rowi]).strip()] in dic_gene_emb:
        gene = dic_snps_gene[snps_i]
        if gene in dic_gene_labeled : #and gene in genes_omim:
            dic_snpsname_emb_sameGENE[snps_i] = 1

        dic_snpsname_emb_un[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])
        dic_snpsname_emb_all[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])
            #dic_snpsname_emb[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])

embedding=np.load(dirr+snps_label_file_embedding_miss)
embedding=np.array(embedding)
df = pd.read_csv(dirr+snps_label_file_miss)
SNPs=list(df["snp"])
for rowi in range(len(SNPs)):
    snps_i = str(SNPs[rowi]).strip()
    dic_snpsname_emb_all[str(SNPs[rowi])] = np.array(embedding[rowi, :])
    if str(SNPs[rowi]) in dic_snps_gene and dic_snps_gene[str(SNPs[rowi]).strip()] in dic_gene_emb:
        gene = dic_snps_gene[snps_i]
        if gene in dic_gene_labeled : #and gene in genes_omim:
            dic_snpsname_emb_sameGENE[snps_i] = 1



print("-------------obtaining UDN snps embedding begin............")
UDN_snps_same_gene=[]
for rowi in range(len(snps_UDN_additional)):
    snps_i = str(snps_UDN_additional[rowi]).strip()
    if  snps_i in dic_snps_gene and dic_snps_gene[snps_i] in dic_gene_emb and snps_i in dic_snpsname_emb_all:
        gene = dic_snps_gene[snps_i]
        if gene in dic_gene_labeled:
            UDN_snps_same_gene.append(snps_i)
            #if gene in genes_omim:
            dic_snpsname_emb_sameGENE[snps_i] = 1


embedding_UDN_shipa=np.array(df_UND[columns[13:]])
print("---embedding_UDN: ",embedding_UDN_shipa.shape)
snps_UDN_shipa=list(df_UND["Name"])
gene_name_UDN_shipa=list(df_UND["gene_name"])
print ("------snps_UDN:  len: ",len(snps_UDN_shipa))

dic_snp_UDN_shipa={}
for rowi in range(len(snps_UDN_shipa)):
    snps_i = str(snps_UDN_shipa[rowi]).strip()
    gene_i=str(gene_name_UDN_shipa[rowi]).strip()
    dic_snpsname_emb_all[snps_i] = np.array(embedding_UDN_shipa[rowi, :])
    if not snps_i in dic_snps_gene:
        dic_snps_gene[snps_i]=gene_i
    if  dic_snps_gene[snps_i] in dic_gene_emb and gene_i in dic_gene_labeled:
        dic_snp_UDN_shipa[snps_i]=1
        dic_snpsname_emb_sameGENE[snps_i] = 1

print("--------dic_snp_UDN_shipa len: ",len(dic_snp_UDN_shipa),"----------in labeled gene with embedding vectors")
        # dic_snpsname_emb_un[str(snps_UDN[rowi]).strip()] = np.array(embedding_UDN[rowi, :])
        # dic_snpsname_emb_all[str(snps_UDN[rowi]).strip()] = np.array(embedding_UDN[rowi, :])
print("-------------obtaining UDN snps embedding end............")




dic_snps_OMIM={}
for snp in dic_snpsname_emb_sameGENE:
    if dic_snps_gene[snp] in genes_OMIM:
        dic_snps_OMIM[snp]=1

dic_snps_OMIM_noClinVar={}
for snp in dic_snpsname_emb_sameGENE:
    if dic_snps_gene[snp] in genes_OMIM:
        dic_snps_OMIM_noClinVar[snp]=1


print("--------------dic_snps_OMIM: ",len(dic_snps_OMIM))
##################predicting GWAS as unlabeled#######################
snps_all_un_prediction=set(list(dic_snpsname_emb_sameGENE.keys()))

snps_all_un_prediction=list(snps_all_un_prediction)
random.shuffle(snps_all_un_prediction)
snps_all_un_prediction=set(snps_all_un_prediction)
print ("------snps_all_un_prediction: all SNPs in the labeled genes and omim to be predicted :  len: ",len(snps_all_un_prediction),"-----------")
if content_unlabel == "ClinVar_all":
    if flag_save_unlabel_predict > 0:
        if not os.path.exists(dirr + model_savename+"_ClinVar_snps_to_predict_all_joint.csv") :
            df = pd.DataFrame({})
            df["snps"] = list(snps_all_un_prediction)
            label_all = [1] * 50000
            label0 = [0] * (abs(len(snps_all_un_prediction) - 50000))
            label_all.extend(label0)

            df["flag"] = label_all[0:len(snps_all_un_prediction)]
            df.to_csv(dirr + model_savename+"_ClinVar_snps_to_predict_all_joint.csv", index=False)
            snps_all_un_prediction = list(snps_all_un_prediction)[0:50000]
            print("-------------ClinVar_predict_all to be predicted : ",len(snps_all_un_prediction), " begining from 0")
        else:
            df = pd.read_csv(dirr + model_savename+"_ClinVar_snps_to_predict_all_joint.csv")
            snps_all_to_predict = list(df["snps"])
            flags = list(df["flag"])
            if 0 in flags:
                index_begin = flags.index(0)
                snps_all_un_prediction = snps_all_to_predict[index_begin:index_begin + 50000]

                for j in range(50000):
                    if index_begin + j < len(flags):
                        flags[index_begin + j] = 1
                df["flag"] = flags
                df.to_csv(dirr +model_savename+"_ClinVar_snps_to_predict_all_joint.csv", index=False)
                print("-------------ClinVar_snps_to_predict_all  to be predicted : ",len(snps_all_to_predict), " begining from : ", index_begin)
            else:
                index_begin=random.randint(0,len(snps_all_to_predict)-50000)
                snps_all_un_prediction = snps_all_to_predict[index_begin:index_begin + 50000]
elif content_unlabel == "UDN_CUI" or  content_unlabel == "UDN_HPO":
    print("UDN_snps_same_gene:  ",len(UDN_snps_same_gene))
    snps_shilpa = set(list(dic_snp_UDN_shipa.keys()))
    snps_all_un_prediction = set(list(dic_snpsname_emb_all.keys()))  # -set(snps_labeled_all)
    snps_shilpa=snps_shilpa.union(set(UDN_snps_same_gene))
    snps_all_un_prediction=snps_all_un_prediction.intersection(snps_shilpa)
    print("-------------snps_all_un_prediction to be predicted UDN CUI and shilpa: ", len(snps_all_un_prediction))
elif "year_" in content_unlabel:
    snps_year = set(list(dic_snp_year.keys()))  # -set(snps_labeled_all)
    snps_all_un_prediction = snps_all_un_prediction.intersection(set(snps_year))
elif "shilpa" in content_unlabel:
    snps_shilpa = set(list(dic_snp_UDN_shipa.keys()))  # -set(snps_labeled_all)
    snps_all_un_prediction = snps_all_un_prediction.intersection(set(snps_shilpa))
    print("-------------snps_all_un_prediction of shilpa to be predicted : ",  len(snps_all_un_prediction))
elif "OMIM" in content_unlabel:
    if "OMIM_noClinVar" in content_unlabel:
        snps_OMIM = set(list(dic_snps_OMIM_noClinVar.keys()))  # -set(snps_labeled_all)
        snps_all_un_prediction = snps_all_un_prediction.intersection(set(snps_OMIM))
        if flag_save_unlabel_predict>0:
            if not os.path.exists(dirr+model_savename+"OMIM_noClinVar_snps_to_predict_joint.csv"):
                df = pd.DataFrame({})
                df["snps"]=list(snps_all_un_prediction)
                label_all=[1]*50000
                label0=[0]*(abs(len(snps_all_un_prediction)-50000))
                label_all.extend(label0)

                df["flag"]=label_all[0:len(snps_all_un_prediction)]
                df.to_csv(dirr+model_savename+"OMIM_noClinVar_snps_to_predict_joint.csv",index=False)
                snps_all_un_prediction=list(snps_all_un_prediction)[0:50000]
                print("-------------snps_all_un_prediction of OMIM and noClinVar to be predicted : ",len(snps_all_un_prediction)," begining from 0")
            else:
                df=pd.read_csv(dirr+model_savename+"OMIM_noClinVar_snps_to_predict_joint.csv")
                snps_all_to_predict=list(df["snps"])
                flags = list(df["flag"])
                if 0 in flags:
                    index_begin=flags.index(0)
                    snps_all_un_prediction=snps_all_to_predict[index_begin:index_begin+50000]

                    for j in range(50000):
                        if index_begin+j<len(flags):
                            flags[index_begin+j]=1
                    df["flag"]=flags
                    df.to_csv(dirr + model_savename+"OMIM_noClinVar_snps_to_predict_joint.csv", index=False)
                    print("-------------snps_all_un_prediction of OMIM and noClinVar to be predicted : ", len(snps_all_to_predict), " begining from : ",index_begin)
                else:
                    index_begin = random.randint(0, len(snps_all_to_predict) - 50000)
                    snps_all_un_prediction = snps_all_to_predict[index_begin:index_begin + 50000]

    else:
        snps_OMIM = set(list(dic_snps_OMIM.keys()))  # -set(snps_labeled_all)
        snps_all_un_prediction = snps_all_un_prediction.intersection(set(snps_OMIM))
        print("-------------snps_all_un_prediction of OMIM to be predicted : ", len(snps_all_un_prediction))


print ("------snps_all_un_prediction: all SNPs in the labeled genes and omim to be predicted :  len: ",len(snps_all_un_prediction),"----------valid----")

###########disease embedding ##################
df_ehr=pd.read_csv(dirr+disease_file_embedding_EHR)
df=pd.read_csv(dirr+disease_file_embedding)
embedding=np.array(df[df.columns[2:]])
embedding_cui_all=[]
print ("CUIs embedding shape: ",embedding.shape)
CUIs = list(pd.read_csv(dirr+disease_file_embedding)["CUIs"])
print (" labeled CUIs: ",len(CUIs))
if not len(CUIs)==len(embedding):
    print ("-----------------------------------------error: ",  "len(CUIs) unequal to len(embedding)--------------------------","error---------------")
for rowi in range(len(CUIs)):
    cui=CUIs[rowi]
    if cui in df_ehr:
        embedding_EHR=np.array(df_ehr[cui])
        dic_cui_emb[str(CUIs[rowi])] = np.concatenate((np.array(embedding[rowi]),embedding_EHR),axis=-1)
        embedding_cui_all.append(np.concatenate((np.array(embedding[rowi]),embedding_EHR),axis=-1))
print ("dic_cui_emb len: ",len(dic_cui_emb))
dic_cui_emb["benign"] = np.min(np.array(embedding_cui_all),axis=0)  #np.zeros(shape=(len(embedding[0, :]),))  #   np.mean(embedding,axis=0)    #


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_loss_CLIP = tf.keras.metrics.Mean(name='train_loss_CLIP')
train_loss_ppi = tf.keras.metrics.Mean(name='train_loss_ppi')
train_ACC = tf.keras.metrics.Accuracy(name='train_ACC')
train_AUC = tf.keras.metrics.AUC(name='train_AUC', curve='ROC')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
VAE_loss = tf.keras.metrics.Mean(name='VAE_loss')
valid_ACC = tf.keras.metrics.Accuracy(name='valid_ACC')
valid_AUC = tf.keras.metrics.AUC(name='valid_AUC', curve='ROC')
valid_F1 = tf.keras.metrics.Mean(name='valid_F1')
valid_PRC = tf.keras.metrics.AUC(name='valid_PRC', curve='PR')


def Model_interaction(input_dim1=768,input_dim2=768+300,kl_weight=weight_kl,latent_dim=64,tau_KL=0.1):
    input1_l = layers.Input(shape=(input_dim1,),dtype=tf.float32)
    input1_l_p = layers.Input(shape=(input_dim1,), dtype=tf.float32)
    input1_l_p_gene = layers.Input(shape=(input_dim1,), dtype=tf.float32)
    input1_l_n = layers.Input(shape=(input_dim1,), dtype=tf.float32)
    input1_l_n_gene = layers.Input(shape=(input_dim1,), dtype=tf.float32)


    input2_l = layers.Input(shape=(input_dim2,),dtype=tf.float32)
    input1_u = layers.Input(shape=(input_dim1,), dtype=tf.float32)
    input1_u_gene = layers.Input(shape=(input_dim1,), dtype=tf.float32)
    input2_u = layers.Input(shape=(input_dim2,), dtype=tf.float32)
    input1_l_gene = layers.Input(shape=(input_dim1,), dtype=tf.float32)
    input_gene_ppi_1 = layers.Input(shape=(input_dim1,), dtype=tf.float32)
    input_gene_ppi_2 = layers.Input(shape=(input_dim1,), dtype=tf.float32)
    encoder1_layer1 = layers.Dense(120, activation=tf.nn.leaky_relu, name="encoder1/fcn1",dtype='float32')
    encoder1_layer1_residual = layers.Dense(120, activation=tf.nn.leaky_relu, name="encoder1/fcn1_residual",dtype='float32')
    encoder1_layer1_residual_h1 = layers.Dense(120, activation=tf.nn.leaky_relu, name="encoder1/fcn1_residual_h1",dtype='float32')

    encoder1_layer2_mean = layers.Dense(int(1.5*latent_dim), activation=tf.nn.leaky_relu, name="encoder1/fcn2/mean",dtype='float32')

    encoder1_layer3= layers.Dense(latent_dim, activation=tf.nn.leaky_relu, name="encoder1/fcn3", dtype='float32')
    encoder1_layer3_h1 = layers.Dense(latent_dim, activation=tf.nn.leaky_relu, name="encoder1/fcn3_h1", dtype='float32')



    encoder2_layer1_1 = layers.Dense(int(1.5*latent_dim), activation=tf.nn.relu, name="encoder2/fcn1",dtype='float32')
    encoder2_layer1_2 = layers.Dense(int(48), activation=tf.nn.relu, name="encoder2/fcn1_2", dtype='float32')
    encoder2_layer2_mean = layers.Dense(int(1.2*latent_dim), activation=tf.nn.leaky_relu, name="encoder2/fcn2/mean",dtype='float32')

    encoder2_layer3 = layers.Dense(latent_dim, activation=tf.nn.leaky_relu, name="encoder2/fcn3", dtype='float32')


    decoder1_layer1 = layers.Dense(int(input_dim1*1.5), activation=tf.nn.leaky_relu, name="decoder1/fcn1",dtype='float32')
    decoder1_layer2 = layers.Dense(input_dim1, activation=None, name="decoder1/fcn2",dtype='float32')

    decoder1_layer1_gene = layers.Dense(int(input_dim1 * 1.5), activation=tf.nn.leaky_relu, name="decoder1/fcn1_gene",
                                   dtype='float32')
    decoder1_layer2_gene = layers.Dense(input_dim1, activation=None, name="decoder1/fcn2_gene", dtype='float32')

    decoder2_layer1_2 = layers.Dense(int(300 * 1.5), activation=tf.nn.leaky_relu, name="decoder2/fcn1_2",dtype='float32')
    decoder2_layer2_2 = layers.Dense(300, activation=None, name="decoder2/fcn2_2", dtype='float32')


    decoder2_layer1 = layers.Dense(int(768*1.5), activation=tf.nn.leaky_relu, name="decoder2/fcn1",dtype='float32')
    decoder2_layer2 = layers.Dense(768, activation=None, name="decoder2/fcn2",dtype='float32')
    # M_interaction = tf.Variable(np.random.normal(0, 1, size=(latent_dim, latent_dim)), trainable=True, dtype=tf.float32, name="interaction_M")
    def AE1(input,input_gene):         ####L2 loss
        feature = encoder1_layer1(input)
        feature_gene = encoder1_layer1(input_gene)
        feature_residual = encoder1_layer1_residual(feature-feature_gene)
        feature=feature+feature_residual
        #feature=feature+encoder1_layer1_residual_h1(feature)
        mean=encoder1_layer2_mean(feature)
        fusioned=mean  #tf.concat([mean,gene_context],axis=1)
        output=decoder1_layer1(mean)  ###using the mean to recontruct
        output=decoder1_layer2(output)
        # output = tf.cast(output, dtype=tf.float32)
        MSE=tf.reduce_mean(tf.square(input*scale_AE-output))
        # output = decoder1_layer_gene(mean)
        output_gene = decoder1_layer1_gene(mean)  ###using the mean to recontruct
        output_gene = decoder1_layer2_gene(output_gene)
        # output = tf.cast(output, dtype=tf.float32)
        MSE_gene = tf.reduce_mean(tf.square(input_gene* scale_AE - output_gene))
        MSE=MSE+MSE_gene
        return fusioned, MSE,mean
    def AE2(input):  ####cross-entropy loss
        feature_1 = encoder2_layer1_1(input[:, 0:768])
        feature_2 = encoder2_layer1_2(input[:, 768:])

        feature = tf.concat([feature_1, feature_2], axis=-1)
        mean = encoder2_layer2_mean(feature)
        output_1 = decoder2_layer1(mean)  ###using the mean to recontruct
        output_1 = decoder2_layer2(output_1)
        output_2 = decoder2_layer1_2(mean)  ###using the mean to recontruct
        output_2 = decoder2_layer2_2(output_2)
        # output = tf.cast(output, dtype=tf.float32)
        MSE1 = tf.reduce_mean(tf.square(input[:, 0:768] * scale_AE - output_1))  # tf.keras.losses.binary_crossentropy(input, output)
        MSE2 = tf.reduce_mean(tf.square(input[:, 768:] * scale_AE - output_2))
        MSE = MSE1 + MSE2
        return mean, MSE, mean  # MSE+kl_weight*kl


    feature1_l_raw, loss_vae1_l, mean1_l = AE1(input1_l,input1_l_gene)
    feature1_l_p, loss_vae1_l_p, mean1_l_p = AE1(input1_l_p, input1_l_p_gene)
    feature1_l_n, loss_vae1_l_n, mean1_l_n = AE1(input1_l_n, input1_l_n_gene)
    feature1_ppi_p, loss_vae1_ppi_p, mean1_ppi_p = AE1(input_gene_ppi_1,input_gene_ppi_2)

    feature2_l_mean, loss_vae2_l, mean2_l = AE2(input2_l)
    feature1_l = encoder1_layer3(feature1_l_raw)
    feature1_l=feature1_l+encoder1_layer3_h1(feature1_l)
    feature2_l = encoder2_layer3(feature2_l_mean)

    similarity_input = tf.matmul(input2_l, input2_l, transpose_b=True) / ((tf.sqrt(tf.reduce_sum(tf.square(input2_l), axis=-1, keepdims=True))) * (
            tf.sqrt(tf.reduce_sum(tf.square(input2_l), axis=-1, keepdims=True))))
    similarity_feature2 = tf.matmul(feature2_l_mean, feature2_l_mean, transpose_b=True) / ((tf.sqrt(tf.reduce_sum(tf.square(feature2_l_mean), axis=-1, keepdims=True))) * (
        tf.sqrt(tf.reduce_sum(tf.square(feature2_l_mean), axis=-1, keepdims=True))))
    print("similarity_feature2 1: ", similarity_feature2)
    similarity_feature2 = tf.nn.softmax(similarity_feature2 / tau_KL, axis=1)
    similarity_input = tf.nn.softmax(similarity_input / tau_KL, axis=1)
    print("similarity_feature2 2: ", similarity_feature2)
    distill_L1 = tf.reduce_mean(tf.reduce_sum(tf.square(similarity_input - similarity_feature2), axis=-1))
    # loss_distill = tf.reduce_mean(tf.reduce_sum(similarity_input * tf.math.log(similarity_input / similarity_feature2), axis=-1) +
    #        tf.reduce_sum(similarity_feature2 * tf.math.log(similarity_feature2 / similarity_input), axis=-1))
    similarity_feature2 = tf.clip_by_value(similarity_feature2, 1e-10, 1.0)
    similarity_input = tf.clip_by_value(similarity_input, 1e-10, 1.0)
    loss_distill = tf.reduce_sum(similarity_feature2 * tf.math.log(similarity_feature2 / similarity_input), axis=-1)

    output = (tf.reduce_sum(tf.multiply(feature1_l, feature2_l), axis=-1)) / (
            tf.sqrt(tf.reduce_sum(tf.square(feature1_l), axis=-1)) * tf.sqrt( tf.reduce_sum(tf.square(feature2_l), axis=-1)))
    # print("------------output: ", output)

    if True:
        feature1_u, loss_vae1_u, mean1_ppi_n = AE1(input1_u,input1_u_gene)
        feature2_u, loss_vae2_u, mean2_u = AE2(input2_u)
        loss_vae = (loss_vae1_l  + loss_vae2_l) / 2.0 +weight_unlabel_snps*(loss_vae1_u  + loss_vae2_u) / 2.0 #+ loss_distill*weight_distill

    model = models.Model(inputs=[input1_l,input2_l, input1_u,input1_u_gene, input2_u,input1_l_gene,input_gene_ppi_1,input_gene_ppi_2,
                                 input1_l_p, input1_l_p_gene, input1_l_n, input1_l_n_gene],
                         outputs=[output,loss_vae, feature1_l,feature2_l,feature1_ppi_p,feature1_u,feature1_l_raw,feature1_l_p,feature1_l_n])
    return model


def train_step(model,epoch_num, input_pro_tsr, input_dis_tsr,label,unlabel_snps,unlabel_snps_gene,
               unlabel_disease,input_gene_emb,input_gene_ppi1_emb,input_gene_ppi2_emb,train_gene_weight,
               input_pro_tsr_positive,input_pro_tsr_positive_gene,input_pro_tsr_negative,input_pro_tsr_negative_gene,train_gene_PPI_p1_weight):

    labels_multi = tf.convert_to_tensor(np.arange(label.numpy().shape[0]))

    train_gene_weight = 1.0 * label.numpy().shape[0] * train_gene_weight / np.sum(train_gene_weight.numpy())
    train_gene_weight = tf.convert_to_tensor(train_gene_weight)
    train_gene_weight = tf.cast(train_gene_weight, dtype=tf.float32)

    batch_size_temp=label.numpy().shape[0]

    with tf.GradientTape(persistent=True) as tape:
        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        prediction,loss_vae,feature_snps,feature_cuis,feature1_l_gene_ppi_p,feature1_l_gene_ppi_n,feature_snp_mean,feature_snp_mean_p,feature_snp_mean_n=\
            model([input_pro_tsr, input_dis_tsr, unlabel_snps,unlabel_snps_gene,unlabel_disease,input_gene_emb,input_gene_ppi1_emb,input_gene_ppi2_emb,
                   input_pro_tsr_positive,input_pro_tsr_positive_gene,input_pro_tsr_negative,input_pro_tsr_negative_gene])

        similarity_snp_p = (tf.reduce_sum(tf.multiply(feature_snp_mean, feature_snp_mean_p), axis=-1)) / (
                    tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean), axis=-1)) * tf.sqrt(
                tf.reduce_sum(tf.square(feature_snp_mean_p), axis=-1)))

        similarity_snp_n = (tf.reduce_sum(tf.multiply(feature_snp_mean, feature_snp_mean_n), axis=-1)) / (
                tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean), axis=-1)) * tf.sqrt(
            tf.reduce_sum(tf.square(feature_snp_mean_n), axis=-1)))

        distance_snp_postive = tf.maximum(0, 0.2 - similarity_snp_p)
        distance_snp_negative = tf.maximum(0, similarity_snp_n - (-0.2))
        loss_snp_contrast = tf.reduce_mean(distance_snp_postive + distance_snp_negative)


        feature_snps = feature_snps / (tf.sqrt(tf.reduce_sum(tf.square(feature_snps), axis=-1, keepdims=True)))
        feature_cuis = feature_cuis / (tf.sqrt(tf.reduce_sum(tf.square(feature_cuis), axis=-1, keepdims=True)))
        feature_interaction = tf.matmul(feature_snps, feature_cuis, transpose_b=True)

        similarity_PPI_p = (tf.reduce_sum(tf.multiply(feature_snp_mean, feature1_l_gene_ppi_p), axis=-1)) / ( tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.square(feature1_l_gene_ppi_p), axis=-1)))

        similarity_PPI_n = (tf.reduce_sum(tf.multiply(feature_snp_mean, feature1_l_gene_ppi_n), axis=-1)) / (
                tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean), axis=-1)) * tf.sqrt( tf.reduce_sum(tf.square(feature1_l_gene_ppi_n), axis=-1)))

        # similarity_PPI_p=tf.reduce_mean(similarity_PPI_p)
        # similarity_PPI_n=tf.reduce_mean(similarity_PPI_n)
        loss_ppi = tf.maximum(similarity_PPI_n + margin_ppi - similarity_PPI_p, 0.0)*train_gene_PPI_p1_weight
        loss_ppi= tf.reduce_mean(loss_ppi)


        if epoch_num<epochs/3:
            tau_softmax_adjust=1.0
        else:
            tau_softmax_adjust=1.0

        if tau_softmax>0.5:
            loss_snps = cce(labels_multi, tf.nn.softmax(tf.transpose(feature_interaction /tau_softmax), axis=-1))*train_gene_weight
            loss_cui = cce(labels_multi, tf.nn.softmax(feature_interaction  / tau_softmax, axis=-1))*train_gene_weight
        else:
            loss_snps = cce(labels_multi, tf.nn.softmax(tf.transpose(feature_interaction* 2.0*tau_softmax_adjust  /tau_softmax), axis=-1))*train_gene_weight
            loss_cui = cce(labels_multi, tf.nn.softmax(feature_interaction * tau_softmax_adjust / tau_softmax, axis=-1))*train_gene_weight



        loss_CLIP_P = loss_snps * label * weight_CLIP_snps + loss_cui * label * weight_CLIP_cui
        #loss_CLIP_N = loss_snps_N * (1 - label)  # * 0.9 + loss_cui_N * (1 - label) * 0.1
        loss_CLIP = tf.reduce_mean(loss_CLIP_P )
        loss_vae = tf.reduce_mean(loss_vae)


        # distance_same = 1 - prediction
        distance_same = tf.maximum( 0,  margin_same- prediction)
        distance_differ = tf.maximum(0, prediction - margin_differ)
        positive_ratio = (batch_size - tf.reduce_sum(label)) / (tf.reduce_sum(label) + 1)
        if weight_cosine>=5:
            loss =  tf.reduce_mean(train_gene_weight* ((batch_size - tf.reduce_sum(label))*label * distance_same*0.99 + (1 - label) * distance_differ*tf.reduce_sum(label) )/batch_size)
        else:
            loss = tf.reduce_mean(train_gene_weight*(1 - label) * distance_differ )

        #loss = tf.reduce_mean( label * distance_same + (1 - label) * distance_differ)
        prediction= tf.nn.sigmoid(prediction)
        loss_vae=tf.reduce_mean(loss_vae)
        if margin_ppi>0:
            weight_ppi=[3.0,3.0,2.5]
        else:
            weight_ppi=[0.0,0.0,0.0]
        if epoch_num < epochs / 3:
            if epoch_num < 6:
                loss_joint =loss_snp_contrast+weight_vae*loss_vae +loss_ppi *  weight_ppi[0]+loss*5.0
            else:
                loss_joint =loss_snp_contrast+weight_vae*loss_vae +loss_ppi *  weight_ppi[1]+loss_CLIP*0.1+loss*5.0
        else:
            loss_joint = weight_vae * loss_vae + loss_CLIP * weight_CLIP + loss_ppi *  weight_ppi[2]+loss_snp_contrast+loss*5.0

    if epoch_num>0:
        gradients = tape.gradient(loss_joint, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    prediction_mean=np.mean(prediction.numpy())
    if prediction_mean==prediction_mean:
        train_loss.update_state(loss)
        VAE_loss.update_state(loss_vae)
        train_loss_CLIP.update_state(loss_CLIP)
        train_loss_ppi.update_state(loss_ppi)
        train_AUC.update_state(label, prediction)
    return loss.numpy(),prediction.numpy(),label.numpy(),

def valid_step(model, input_pro_tsr, input_dis_tsr, label,input_gene_emb):
    prediction,loss_vae,feature_snps,feature_cuis,feature_ppi_1 ,feature_ppi_2,feature_snp_mean,feature_snp_mean_p,feature_snp_mean_n= \
        model([input_pro_tsr, input_dis_tsr, input_pro_tsr, input_gene_emb,input_dis_tsr,
               input_gene_emb,input_pro_tsr,input_gene_emb,input_pro_tsr,input_pro_tsr,input_pro_tsr,input_pro_tsr])
    # prediction = tf.nn.sigmoid(prediction)
    valid_AUC.update_state(label, tf.nn.sigmoid(prediction))
    valid_PRC.update_state(label, tf.nn.sigmoid(prediction))


    return label.numpy(), prediction.numpy()

def train_model(model,ds_train, ds_test, eval_SNPs,eval_index,epochs,eval_SNPs_train,eval_cui_train,eval_gene_train,eval_gene_test):
    print("--------begin training.....")
    epoch_num = -1
    auc_total = []
    save_flag=False
    #########################################for test######################################
    while (epoch_num < epochs):
        if epoch_num%epoch_MRR==10:
            MRR=[]
            rank_total=[]
            dic_snps_recall10={}
            dic_snps_recall50={}
            print("-------------------------------------------------evaluate the top k accuracy-----------------------------------")

            CUIs_all_covered=list(pd.read_csv(dirr+file_CUIs_target)["CUIs"])
            print ("---CUIs_all_covered: ",len(CUIs_all_covered))

            CUIs_embedding_test = []
            add_num = batch_size - len(CUIs_all_covered) % batch_size
            batch_size_test = len(CUIs_all_covered) + add_num
            for cui in CUIs_all_covered:
                CUIs_embedding_test.append(dic_cui_emb[str(cui)])
            for addi in range(add_num):
                CUIs_embedding_test.append(CUIs_embedding_test[random.randint(0, batch_size)])
            CUIs_embedding_test = np.array(CUIs_embedding_test)
            CUIs_embedding_test_input = tf.convert_to_tensor(CUIs_embedding_test, dtype=tf.float32)
            snps_test = 0


            batch_max = int(len(CUIs_embedding_test_input) / batch_size)
            for snps_i in range(len(eval_SNPs)):
                snps_index = eval_SNPs[snps_i]
                snps = dic_index_snps[snps_index]

                pair_snps_cui = str(snps_index) + "_snps_cui_" + str(eval_index[snps_i])
                dic_snps_recall10[pair_snps_cui] = 0
                dic_snps_recall50[pair_snps_cui] = 0

                # dic_snps_recall10[snps_index] = 0
                # dic_snps_recall50[snps_index] = 0
                snps_test += 1
                embedding_snps = dic_snpsname_emb_all[snps]
                embedding_snps = np.tile(embedding_snps, (batch_size_test, 1))
                embedding_snps_input = tf.convert_to_tensor(embedding_snps, dtype=tf.float32)

                embedding_gene = dic_gene_emb[dic_snps_gene[snps]]
                embedding_gene = np.tile(embedding_gene, (batch_size_test, 1))
                embedding_gene_input = tf.convert_to_tensor(embedding_gene, dtype=tf.float32)


                prediction_all = []
                feature_snps_all = []
                feature_cuis_all = []
                for batch_i in range(batch_max):
                    prediction, loss_vae, feature_snps, feature_cuis,feature_temp1_,feature_temp2_,feature_snp_mean,_,_ = model(
                        [embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         CUIs_embedding_test_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         CUIs_embedding_test_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                         embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :]])
                    prediction_all.append(prediction.numpy())
                    feature_snps_all.append(feature_snps.numpy())
                    feature_cuis_all.append(feature_cuis.numpy())
                prediction_all = np.array(prediction_all).reshape((batch_size * batch_max, -1))
                prediction_all = list(prediction_all)[0:len(CUIs_all_covered)]  # tf.nn.sigmoid(prediction)

                index_ranking = len(prediction_all) + 1 - ss.rankdata(prediction_all, method='max')

                MRR.append(1.0 / index_ranking[eval_index[snps_i]])
                rank_total.append(index_ranking[eval_index[snps_i]])

                if index_ranking[eval_index[snps_i]] < 51:
                    dic_snps_recall50[pair_snps_cui] = dic_snps_recall50[pair_snps_cui] + 1
                    if index_ranking[eval_index[snps_i]] < 11:
                        dic_snps_recall10[pair_snps_cui] = dic_snps_recall10[pair_snps_cui] + 1

            MRR=np.mean(MRR)
            recall50=1.0*np.sum(list(dic_snps_recall50.values()))/len(eval_SNPs)
            recall10 = 1.0 * np.sum(list(dic_snps_recall10.values())) / len(eval_SNPs)
            print("---------------MRR: ", MRR, "recall10: ", recall10, "recall50: ", recall50, "rank_mean: ",
                  np.mean(rank_total), "rank_median: ", np.median(rank_total))
        #########################################for test######################################
        epoch_num += 1
        train_label = []
        train_prediction = []
        i_number = -1
        except_number=0
        for traindata_snps,traindata_cuis, traindata_Y,unlabel_snps,unlabel_snps_gene, unlabel_disease,traindata_gene,traindata_gene_ppi_1, traindata_gene_ppi_2, train_gene_weight, \
                traindata_snps_positive,traindata_snps_positive_gene,traindata_snps_negative,traindata_snps_negative_gene,train_gene_PPI_p1_weight in ds_train:
            i_number += 1
            if True:
                loss,prediction, label =  \
                    train_step(model, epoch_num,traindata_snps, traindata_cuis,traindata_Y, unlabel_snps,unlabel_snps_gene,
                               unlabel_disease,traindata_gene,traindata_gene_ppi_1, traindata_gene_ppi_2,train_gene_weight,
                               traindata_snps_positive,traindata_snps_positive_gene,traindata_snps_negative,traindata_snps_negative_gene,train_gene_PPI_p1_weight)

            if i_number == 0:
                train_prediction = np.array(prediction)
                train_label = np.array(label)
            else:
                train_prediction = np.concatenate((train_prediction, np.array(prediction)), axis=0)
                train_label = np.concatenate((train_label, np.array(label)), axis=0)

        AUC_overall_train = roc_auc_score(train_label, train_prediction)
        if epoch_num % epoch_show == 0 or epoch_num > epochs - 3:
            label_valid = []
            pred_valid = []
            snps_valid=[]
            pair_valid=[]
            dic_cui_prediction={}
            dic_cui_snps={}
            dic_cui_label={}
            dic_snps_prediction = {}
            dic_snps_cuis = {}
            dic_snps_label = {}
            i_number = -1

            for testdata_cuis,testdata_snps,testdata_Y,test_names,test_pair,testdata_cuis_gene in ds_test:
                i_number+=1
                label, prediction = valid_step(model, testdata_snps, testdata_cuis, testdata_Y,testdata_cuis_gene)
                if i_number == 0:
                    pred_valid = np.array(prediction)
                    label_valid = np.array(testdata_Y.numpy())
                    snps_valid = np.array(test_names.numpy())
                    pair_valid = np.array(test_pair.numpy())
                else:
                    pred_valid = np.concatenate((pred_valid, np.array(prediction)), axis=0)
                    label_valid = np.concatenate((label_valid, np.array(testdata_Y.numpy())), axis=0)
                    snps_valid = np.concatenate((snps_valid, np.array(test_names.numpy())), axis=0)
                    pair_valid = np.concatenate((pair_valid, np.array(test_pair.numpy())), axis=0)

            pred_valid=np.array(pred_valid).reshape((-1, 1))
            label_valid = np.array(label_valid).reshape((-1, 1))
            snps_valid = np.array(snps_valid).reshape((-1, 1))
            pair_valid = np.array(pair_valid).reshape((-1, 1))

            label_all=[]
            prediction_all=[]

            CUIs_val=[]
            SNPs_val=[]
            prediction_val=[]
            label_val=[]
            for rowi in range(len(pred_valid)):
                snps=str(pair_valid[rowi]).split("_")[0]
                cui = str(pair_valid[rowi]).split("_")[1]
                prediction = pred_valid[rowi][0]
                label = label_valid[rowi][0]

                SNPs_val.append(snps_valid[rowi][0])
                CUIs_val.append(cui[0:-2])
                prediction_val.append(prediction)
                label_val.append(label)

                label_all.append(label)
                prediction_all.append(prediction)

                dic_cui_prediction.setdefault(cui,[]).append(prediction)
                dic_cui_label.setdefault(cui, []).append(label)

                dic_snps_prediction.setdefault(snps_valid[rowi][0], []).append(prediction)
                dic_snps_label.setdefault(snps_valid[rowi][0], []).append(label)

            df = pd.DataFrame({})
            print("-------------saving SNP_CUI_score_label_test: len prediction_val: ",len(prediction_val))
            df["SNPs"] = SNPs_val
            df["CUI"] = CUIs_val
            df["score"] = prediction_val
            df["label"] = label_val
            df.to_csv(dirr_results + "SNP_CUI_score_label_test.csv", index=False)

            AUC_SNPs=[]
            AUC_SNPs_name=[]
            AUC_cuis=[]
            AUC_cui_name=[]

            PRC_gain_SNPs=[]
            PRC_gain_cuis=[]
            PRC_SNPs=[]

            PRC_cuis=[]
            PRC_cui_name=[]
            PRC_SNPs_name = []
            for snps in dic_snps_prediction:
                if np.sum(dic_snps_label[snps]) > 0 and not np.sum(dic_snps_label[snps])==len(dic_snps_label[snps]):
                    # print ("dic_snps_label[snps]: ",len(dic_snps_label[snps]))
                    # print("dic_snps_prediction[snps]: ", len(dic_snps_prediction[snps]))
                    auc = roc_auc_score(dic_snps_label[snps], dic_snps_prediction[snps])
                    AUC_SNPs.append(auc)
                    AUC_SNPs_name.append(snps)

                    lr_precision, lr_recall, _ = precision_recall_curve(dic_snps_label[snps], dic_snps_prediction[snps])
                    prc = metrics.auc(lr_recall, lr_precision)
                    Prevelance = 1.0 * np.sum(dic_snps_label[snps]) / len(dic_snps_label[snps])
                    prc_gain = 1.0 * (prc - Prevelance) / Prevelance
                    PRC_gain_SNPs.append(prc_gain)
                    PRC_SNPs.append(prc)
                    PRC_SNPs_name.append(snps)

            for cui in dic_cui_prediction:
                if np.sum(dic_cui_label[cui])>0 and not np.sum(dic_cui_label[cui])==len(dic_cui_label[cui]):
                    auc = roc_auc_score(dic_cui_label[cui], dic_cui_prediction[cui])
                    AUC_cuis.append(auc)
                    AUC_cui_name.append(cui)
                    lr_precision, lr_recall, _ = precision_recall_curve(dic_cui_label[cui], dic_cui_prediction[cui])
                    prc = metrics.auc(lr_recall, lr_precision)
                    Prevelance = 1.0 * np.sum(dic_cui_label[cui]) / len(dic_cui_label[cui])
                    prc_gain = 1.0 * (prc - Prevelance) / Prevelance
                    PRC_gain_cuis.append(prc_gain)
                    PRC_cuis.append(prc)
                    PRC_cui_name.append(cui)

            df = pd.DataFrame({})
            df["AUC"] = AUC_cuis
            df["CUI"] = AUC_cui_name
            df.to_csv(dirr_results + "AUC_CUI_test.csv", index=False)

            df = pd.DataFrame({})
            df["AUC"] = AUC_SNPs
            df["SNP"] = AUC_SNPs_name
            df.to_csv(dirr_results + "AUC_SNPs_test.csv", index=False)

            df = pd.DataFrame({})
            df["PRC"] = PRC_cuis
            df["CUI"] = PRC_cui_name
            df.to_csv(dirr_results + "PRC_CUI_test.csv", index=False)

            df = pd.DataFrame({})
            df["PRC"] = PRC_SNPs
            df["SNP"] = PRC_SNPs_name
            df.to_csv(dirr_results + "PRC_SNPs_test.csv", index=False)

            if not len(prediction_all)==np.sum(label_all) and np.sum(label_all)>0:
                AUC_overall = roc_auc_score(label_all, prediction_all)
                auc_total.append(np.mean(AUC_overall))
            else:
                AUC_overall=0.5
            lr_precision, lr_recall, _ = precision_recall_curve(label_all, prediction_all)
            PRC_overall = metrics.auc(lr_recall, lr_precision)
            Prevelance_overall=1.0*np.sum(label_all)/len(label_all)
            PRC_overall_gain=1.0*(PRC_overall-Prevelance_overall)/Prevelance_overall

            print ("PRC_overall: ",PRC_overall ," prevelance: ",Prevelance_overall, "prevelance gain: ",PRC_overall_gain )

            if epoch_num>10:
               if np.mean(auc_total[-8:-4])>np.mean(auc_total[-4:]):
                   epoch_num+=1
            print("---epoch: %i,  loss: %3f, VAE_loss: %3f, CLIP_loss: %3f, train_loss_ppi: %3f, train_AUC: %3f, auc_overall: %3f,AUC_SNPs: %3f, "
                  "AUC_disease: %3f prc_gain: %3f , prc_gain_snps: %3f, prc_gain_disease: %3f " % (epoch_num, train_loss.result(),VAE_loss.result(),train_loss_CLIP.result(),train_loss_ppi.result(),
                     AUC_overall_train,  np.mean(AUC_overall), np.mean(AUC_SNPs),np.mean(AUC_cuis),
                     PRC_overall_gain,np.mean(PRC_gain_SNPs),np.mean(PRC_gain_cuis)))

            if save_flag == False and epoch_num > epochs - 3:
                MRR = []
                dic_snps_recall10 = {}
                dic_snps_recall50 = {}
                rank_total=[]
                print( "-------------------------------------------------begin saving predictions-----------------------------------")
                save_flag = True
                CUIs_all_covered = list(pd.read_csv(dirr + file_CUIs_target)["CUIs"])
                print("---CUIs_all_covered: ", len(CUIs_all_covered))

                CUIs_embedding_test = []
                add_num = batch_size - len(CUIs_all_covered) % batch_size
                batch_size_test = len(CUIs_all_covered) + add_num
                for cui in CUIs_all_covered:
                    CUIs_embedding_test.append(dic_cui_emb[str(cui)])
                for addi in range(add_num):
                    CUIs_embedding_test.append(CUIs_embedding_test[random.randint(0, batch_size)])
                CUIs_embedding_test = np.array(CUIs_embedding_test)
                CUIs_embedding_test_input = tf.convert_to_tensor(CUIs_embedding_test, dtype=tf.float32)
                snps_test = 0

                feature_snps_save=[]
                feature_cuis_save=[]
                cuiname_save=[]
                snpsname_save=[]
                batch_max = int(len(CUIs_embedding_test_input) / batch_size)
                MRR_save=[]
                MRR_SNP=[]
                MRR_CUI=[]
                Ranking_save=[]

                for snps_i in range(len(eval_SNPs)):
                    snps_index = eval_SNPs[snps_i]
                    snps = dic_index_snps[snps_index]

                    pair_snps_cui = str(snps_index) + "_snps_cui_" + str(eval_index[snps_i])
                    dic_snps_recall10[pair_snps_cui] = 0
                    dic_snps_recall50[pair_snps_cui] = 0

                    #dic_snps_recall10[snps_index] = 0
                    #dic_snps_recall50[snps_index] = 0
                    snps_test += 1
                    embedding_snps = dic_snpsname_emb_all[snps]
                    embedding_snps = np.tile(embedding_snps, (batch_size_test, 1))
                    embedding_snps_input = tf.convert_to_tensor(embedding_snps, dtype=tf.float32)

                    embedding_gene = dic_gene_emb[dic_snps_gene[snps]]
                    embedding_gene = np.tile(embedding_gene, (batch_size_test, 1))
                    embedding_gene_input = tf.convert_to_tensor(embedding_gene, dtype=tf.float32)

                    prediction_all = []
                    feature_snps_all = []
                    feature_cuis_all = []

                    for batch_i in range(batch_max):
                        prediction, loss_vae, feature_snps, feature_cuis,feature_temp1_,feature_temp2_,feature_snp_mean,_,_ = model(
                            [embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             CUIs_embedding_test_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             CUIs_embedding_test_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :]])
                        prediction_all.append(prediction.numpy())
                        feature_snps_all.append(feature_snps.numpy())
                        feature_cuis_all.append(feature_cuis.numpy())
                    prediction_all = np.array(prediction_all).reshape((batch_size * batch_max, -1))
                    prediction_all = list(prediction_all)[0:len(CUIs_all_covered)]  # tf.nn.sigmoid(prediction)
                    index_ranking = len(prediction_all) + 1 - ss.rankdata(prediction_all, method='dense')
                    MRR.append(1.0 / index_ranking[eval_index[snps_i]])

                    MRR_save.append(1.0 / index_ranking[eval_index[snps_i]])
                    MRR_SNP.append(snps)
                    MRR_CUI.append(CUIs_all_covered[eval_index[snps_i]])
                    Ranking_save.append(index_ranking[eval_index[snps_i]])

                    rank_total.append(index_ranking[eval_index[snps_i]])
                    feature_snps_all = np.array(feature_snps_all).reshape((batch_size * batch_max, -1))
                    feature_cuis_all = np.array(feature_cuis_all).reshape((batch_size * batch_max, -1))
                    feature_snps_save.append(feature_snps_all[0])
                    feature_cuis_save.append(feature_cuis_all[eval_index[snps_i]])
                    cuiname_save.append(CUIs_all_covered[eval_index[snps_i]])
                    snpsname_save.append(snps)
                    if index_ranking[eval_index[snps_i]] < 51:
                        dic_snps_recall50[pair_snps_cui] = dic_snps_recall50[pair_snps_cui] + 1
                        if index_ranking[eval_index[snps_i]] < 11:
                            dic_snps_recall10[pair_snps_cui] = dic_snps_recall10[pair_snps_cui] + 1

                MRR = np.mean(MRR)
                recall50 = 1.0 * np.sum(list(dic_snps_recall50.values())) / len(eval_SNPs)
                recall10 = 1.0 * np.sum(list(dic_snps_recall10.values())) / len(eval_SNPs)
                print("---------------MRR: ", MRR, "recall10: ", recall10, "recall50: ", recall50, "rank_mean: ",np.mean(rank_total), "rank_median: ",np.median(rank_total))

                f = open(dirr_results_main + filename_eval, 'a')
                f.write("Time_pre: %2f min, weight_vae: %2f, AUC_train :%4f, AUC_overall:%4f, AUC_snps:%4f, AUC_disease:%4f,"
                            " prc_overall :%4f, prc_overall_gain :%4f, PRC_snps :%4f, PRC_disease:%4f,"
                        " MRR:%4f, recall10:%4f, recall50:%4f , rank_mean:%2f, rank_median:%2f " %
                        (time_preprocessing, weight_vae,AUC_overall_train, np.mean(AUC_overall), np.mean(AUC_SNPs),  np.mean(AUC_cuis),
                                PRC_overall, PRC_overall_gain, np.mean(PRC_SNPs),  np.mean(PRC_cuis), MRR,recall10,recall50,np.mean(rank_total),np.median(rank_total)))
                f.write("\r")
                f.close()
                print("------------------------------------------------- saving predictions end-----------------------------------")

                np.save(dirr_results + "feature_test_snps", np.array(feature_snps_save))
                np.save(dirr_results + "feature_test_disease", np.array(feature_cuis_save))
                df = pd.DataFrame({})
                df["MRR"] = MRR_save
                df["snps"] = MRR_SNP
                df["CUI"]=MRR_CUI
                df["rank"]=Ranking_save
                df.to_csv(dirr_results + "MRR_snps.csv", index=False)

                df=pd.DataFrame({})
                df["snps"]=snpsname_save
                df["cui"]=cuiname_save
                df.to_csv(dirr_results+"snps_cuis_names_test.csv",index=False)
                ##################################
                print ("--------------------begin saving training features of snps-dsiease paris-------------------------------")
                CUIs_embedding_train = []
                snps_embedding_train=[]
                gene_embedding_train=[]
                cuiname_save = []
                snpsname_save = []
                for snps_index,cui,emb_gene in zip(eval_SNPs_train,eval_cui_train,eval_gene_train):
                    snps = dic_index_snps[snps_index]
                    snps_embedding_train.append(dic_snpsname_emb_all[snps])
                    CUIs_embedding_train.append(dic_cui_emb[str(cui)])
                    gene_embedding_train.append(emb_gene)
                    cuiname_save.append(cui)
                    snpsname_save.append(snps)
                for i in range(batch_size*1):
                    index_i=random.randint(0,len(snps_embedding_train)-1)
                    snps_embedding_train.append(snps_embedding_train[index_i])
                    CUIs_embedding_train.append(CUIs_embedding_train[index_i])
                    gene_embedding_train.append(gene_embedding_train[index_i])
                    cuiname_save.append(cuiname_save[index_i])
                    snpsname_save.append(snpsname_save[index_i])

                snps_embedding_train = np.array(snps_embedding_train)
                snps_embedding_train = tf.convert_to_tensor(snps_embedding_train, dtype=tf.float32)
                CUIs_embedding_train = np.array(CUIs_embedding_train)
                CUIs_embedding_train = tf.convert_to_tensor(CUIs_embedding_train, dtype=tf.float32)
                gene_embedding_train = np.array(gene_embedding_train)
                gene_embedding_train = tf.convert_to_tensor(gene_embedding_train, dtype=tf.float32)

                feature_snps_save=[]
                feature_cuis_save=[]
                batch_max=int(len(CUIs_embedding_train)/batch_size)-2
                for batch_i in range(batch_max):
                    prediction, loss_vae, feature_snps, feature_cuis,feature_temp1_,feature_temp2_,feature_snp_mean,_,_ =\
                        model( [snps_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                CUIs_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                             snps_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                gene_embedding_train[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                             CUIs_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                            gene_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                snps_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                gene_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                snps_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                snps_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                snps_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:],
                                snps_embedding_train[batch_i*batch_size:(batch_i+1)*batch_size,:]])
                    feature_snps_save.append(feature_snps)
                    feature_cuis_save.append(feature_cuis)
                feature_snps_save=np.array(feature_snps_save).reshape((batch_size*batch_max,latent_dim))
                feature_cuis_save = np.array(feature_cuis_save).reshape((batch_size * batch_max, latent_dim))
                np.save(dirr_results + "feature_train_snps", np.array(feature_snps_save))
                np.save(dirr_results + "feature_train_disease", np.array(feature_cuis_save))

                df = pd.DataFrame({})
                df["snps"] = snpsname_save[0:batch_size*batch_max]
                df["cui"] = cuiname_save[0:batch_size*batch_max]
                df.to_csv(dirr_results + "snps_cuis_names_train.csv", index=False)
                print("--------------------end saving training features of snps-dsiease paris-------------------------------")

                ##################################
                if flag_save_unlabel_emb>0:
                    print( "--------------------begin saving unlabeled  snps-------------------------------")
                    CUIs_embedding_train = []
                    snps_embedding_train = []
                    snpsname_un=[]
                    cuis_all = list(pd.read_csv(dirr + file_CUIs_target)["CUIs"])
                    snps_all=list(snps_all_un_prediction)#  list(set(list(dic_snpsname_emb_un.keys()))-set(snps_labeled_all))
                    gene_embedding_train=[]
                    print("snps_all to be predicted: ",len(snps_all))

                    # snps_all=list(set(snps_all)-set(eval_SNPs)-set(eval_SNPs_train))
                    for samplei in range(len(snps_all)):
                        index_cui=random.randint(0,len(cuis_all)-1)
                        #index_snps =samplei# random.randint(0, len(snps_all) - 1)
                        snps=snps_all[samplei]
                        if snps in dic_snps_gene:
                            gene=dic_snps_gene[snps]
                            snps_embedding_train.append(dic_snpsname_emb_all[snps])
                            CUIs_embedding_train.append(dic_cui_emb[cuis_all[index_cui]])
                            gene_embedding_train.append(dic_gene_emb[gene])
                            snpsname_un.append(snps)
                    snps_embedding_train = np.array(snps_embedding_train)
                    snps_embedding_train = tf.convert_to_tensor(snps_embedding_train, dtype=tf.float32)
                    CUIs_embedding_train = np.array(CUIs_embedding_train)
                    CUIs_embedding_train = tf.convert_to_tensor(CUIs_embedding_train, dtype=tf.float32)
                    gene_embedding_train = np.array(gene_embedding_train)
                    gene_embedding_train = tf.convert_to_tensor(gene_embedding_train, dtype=tf.float32)

                    feature_snps_save = []
                    batch_max = int(len(CUIs_embedding_train) / batch_size)-1
                    for batch_i in range(batch_max):
                        prediction, loss_vae, feature_snps, feature_cuis , feature_temp1_, feature_temp2_,feature_snp_mean,_,_= model(
                            [snps_embedding_train[batch_i * batch_size:(batch_i + 1) * batch_size,:],
                             CUIs_embedding_train[batch_i * batch_size:(batch_i + 1) * batch_size,:],
                             snps_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:],
                             gene_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size, :],
                             CUIs_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:],
                             gene_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:],
                             snps_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:],
                             gene_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:],
                             snps_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:],
                             snps_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:],
                             snps_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:],
                             snps_embedding_train[batch_i * batch_size: (batch_i + 1) * batch_size,:]])
                        feature_snps_save.append(feature_snps)
                    feature_snps_save = np.array(feature_snps_save).reshape((batch_size * batch_max, latent_dim))
                    print ("feature_snps_save shape: ",feature_snps_save.shape)
                    np.save(dirr_results + "feature_unlabel_snps", np.array(feature_snps_save))

                    df = pd.DataFrame({})
                    df["snps"] = snpsname_un[0:batch_size * batch_max]
                    df.to_csv(dirr_results + "snps_names_unlabel.csv", index=False)

                if flag_save_unlabel_predict > 0:
                    print(
                        "-------------------- begin saving unlabeled  snps  predictions-------------------------------")
                    if content_unlabel == "UDN_CUI":
                        CUIs_all_covered = list(pd.read_csv(dirr + file_CUIs_target)["CUIs"])

                    elif content_unlabel == "UDN_HPO":
                        CUIs_all_covered = list(pd.read_csv(dirr + file_CUIs_target_UDN)["HPO"])
                    else:
                        CUIs_all_covered = list(pd.read_csv(dirr +  file_CUIs_target)["CUIs"])


                    CUIs_embedding_test = []
                    add_num= batch_size-len(CUIs_all_covered)%batch_size
                    batch_size_test = len(CUIs_all_covered)+add_num
                    for cui in CUIs_all_covered:
                        CUIs_embedding_test.append(dic_cui_emb[str(cui)])
                    for addi in range(add_num):
                        CUIs_embedding_test.append(CUIs_embedding_test[random.randint(0,batch_size)])
                    CUIs_embedding_test = np.array(CUIs_embedding_test)
                    CUIs_embedding_test_input = tf.convert_to_tensor(CUIs_embedding_test, dtype=tf.float32)
                    snps_all = list(snps_all_un_prediction)  #
                    snps_num_predict = 0
                    feature_snps_save = []
                    feature_disease_save=[]
                    snps_save_all = []
                    data_save_all = []
                    batch_max = int(batch_size_test / batch_size)
                    CUI_save_all = []
                    score_save_all = []

                    dic_snp_score = {}
                    dic_snp_cuis = {}
                    top_N=100
                    if  content_unlabel == "UDN_CUI" or  content_unlabel == "UDN_HPO" or "shilpa" in content_unlabel or "year" in content_unlabel or "OMIM" in content_unlabel:
                        top_N=len(CUIs_all_covered)

                    top_N = len(CUIs_all_covered)
                    for snps in snps_all[0:50000]:
                        if snps==snps and snps in dic_snpsname_emb_all:
                            snps_num_predict += 1
                            embedding_snps = dic_snpsname_emb_all[snps]
                            embedding_snps = np.tile(embedding_snps, (batch_size_test, 1))
                            embedding_snps_input = tf.convert_to_tensor(embedding_snps, dtype=tf.float32)
                            embedding_gene = dic_gene_emb[dic_snps_gene[snps]]
                            embedding_gene = np.tile(embedding_gene, (batch_size_test, 1))
                            embedding_gene_input = tf.convert_to_tensor(embedding_gene, dtype=tf.float32)
                            prediction_all = []
                            feature_snps_all = []
                            feature_cuis_all = []
                            for batch_i in range(batch_max):
                                prediction, loss_vae, feature_snps, feature_cuis,feature_temp1_,feature_temp2_,feature_snp_mean,_,_ = model(
                                    [embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     CUIs_embedding_test_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     CUIs_embedding_test_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_gene_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :],
                                     embedding_snps_input[batch_i * batch_size:(batch_i + 1) * batch_size, :]])
                                prediction_all.append(prediction.numpy())
                                feature_snps_all.append(feature_snps.numpy())
                                feature_cuis_all.append(feature_cuis.numpy())
                            prediction_all = np.array(prediction_all).reshape((batch_size * batch_max, -1))
                            prediction_all = list(prediction_all)[0:len(CUIs_all_covered)]  # tf.nn.sigmoid(prediction)

                            feature_snps_all = np.array(feature_snps_all).reshape((batch_size * batch_max, -1))
                            feature_cuis_all = np.array(feature_cuis_all).reshape((batch_size * batch_max, -1))

                            #feature_disease_save = feature_cuis_all[0:len(CUIs_all_covered), :]
                            feature_snps_save.append(feature_snps_all[0])

                            prediction_sort, CUIs_all_covered_sort = zip(*sorted(zip(prediction_all, CUIs_all_covered), reverse=True))

                            CUI_save_all.append(CUIs_all_covered_sort[0:top_N])
                            score_save_all.append(prediction_sort[0:top_N])
                            snps_save_all.append(snps)
                            dic_snp_score[snps] = prediction_sort[0:top_N]
                            dic_snp_cuis[snps] = CUIs_all_covered_sort[0:top_N]

                            if True:  # content_unlabel=="ALL":
                                if  snps_num_predict % 50000 == 0 or snps_num_predict == len(snps_all):
                                    print("-------------snps_num_predict: ", snps_num_predict)
                                    savename_unlabel_predict_final = str(
                                        snps_num_predict) + "_CUI_" + savename_unlabel_predict
                                    CUI_save_all = np.array(CUI_save_all).T
                                    #np.save(dirr_results + savename_unlabel_predict_final, np.array(CUI_save_all[0:snps_num_predict]).T)
                                    df = pd.DataFrame(CUI_save_all)
                                    df.to_csv(dirr_results + savename_unlabel_predict_final, header=snps_save_all, index=False)

                                    savename_unlabel_predict_final = str( snps_num_predict) + "_score_" + savename_unlabel_predict
                                    score_save_all = np.array(score_save_all).T
                                    score_save_all = np.squeeze(score_save_all)
                                    #np.save(dirr_results + savename_unlabel_predict_final, np.array(score_save_all[0:snps_num_predict]).T)
                                    df = pd.DataFrame(score_save_all)
                                    df.to_csv(dirr_results + savename_unlabel_predict_final, header=snps_save_all,index=False)


                                    np.save(dirr_results + "feature_predicted_snps", np.array(feature_snps_save))
                                    df = pd.DataFrame({})
                                    df["snps"] = snps_all[0:snps_num_predict]
                                    df.to_csv(dirr_results + "snps_names_predicted_snps.csv", index=False)
                    print("-------------------- end saving unlabeled  snps  predictions-------------------------------")
        train_loss.reset_states()
        train_AUC.reset_states()
        VAE_loss.reset_states()
        train_loss_CLIP.reset_states()
        train_loss_ppi.reset_states()

if __name__ == '__main__':
    if not os.path.exists(dirr_save_model):
        os.makedirs(dirr_save_model)
    model = Model_interaction(latent_dim=latent_dim)
    savename_model =  dirr_save_model+model_savename + "_model"
    if os.path.exists(savename_model) and flag_reload > 0:
        print("---------------------------------------------loadding saved model....................")
        model = tf.keras.models.load_model(savename_model)
    print("loadding data....................")
    (traindata_snps,traindata_cuis,traindata_snps_P,traindata_cuis_P,traindata_gene_P,traindata_Y,traindata_names,
     testdata_cuis,testdata_snps,testdata_Y,testdata_names,test_pair,eval_SNPs,eval_index,
     unlabel_snps,unlabel_gene,unlabel_disease,eval_SNPs_train,eval_cui_train,train_gene,train_gene_ppi_1, train_gene_ppi_2,test_gene,eval_gene_train,eval_gene_test,train_gene_weight,
     traindata_snps_positive,traindata_snps_positive_gene,traindata_snps_negative,traindata_snps_negative_gene,train_gene_PPI_p1_weight)=\
        loaddata(train_snps_ratio=train_ratio,negative_disease=negative_disease,
                 negative_snps=negative_snps,flag_hard_mining=flag_hard_negative,
                 snps_remove=snps_all_un_prediction,flag_debug=flag_debug,
                 flag_negative_filter=flag_negative_filter,flag_cross_gene=flag_cross_gene,flag_cross_cui=flag_cross_cui)
    start = time.process_time()
    ds_test = tf.data.Dataset.from_tensor_slices(
        (testdata_cuis, testdata_snps, testdata_Y, testdata_names, test_pair, test_gene)). \
        shuffle(buffer_size=batch_size * 1).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

    ds_train = tf.data.Dataset.from_tensor_slices((traindata_snps,traindata_cuis,traindata_Y,unlabel_snps,unlabel_gene,
                                                   unlabel_disease,train_gene,train_gene_ppi_1, train_gene_ppi_2,train_gene_weight,
                                                   traindata_snps_positive,traindata_snps_positive_gene,traindata_snps_negative,traindata_snps_negative_gene,train_gene_PPI_p1_weight)).\
        shuffle(buffer_size=batch_size*30).batch(batch_size).\
        prefetch(tf.data.experimental.AUTOTUNE).cache()
    end = time.process_time()
    print("---Elapsed time for pre-processing train and test by tensorflow", (end - start) /60.0, " min.")
    time_preprocessing=(end - start) /60.0
    print("model training begin....")
    train_model(model, ds_train,ds_test,eval_SNPs,eval_index,epochs,eval_SNPs_train,eval_cui_train,eval_gene_train,eval_gene_test)
    print("model training end....")
    if flag_modelsave>0:
        model.save(savename_model)
    print("model saving end....")

