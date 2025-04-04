#!/usr/bin/env python
# coding: utf-8
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
from utilization import loaddata
import scipy.stats as ss
import time
import argparse
tf.keras.backend.set_floatx('float32')
# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--main_dirr', type=str, default="data/",
                        help='main data directory')
    parser.add_argument('--file_CUIs_target', type=str, default="CUIs_all_covered_CODER_4179.csv",
                        help='file including target CUIs')
    parser.add_argument('--file_annotations', type=str, default="Clivar_snps_disease_Pathogenic_Likely_4_11_subset.csv",
                        help='0-1 to denote if reload the model')
    parser.add_argument('--file_snps_labeled', type=str, default="Binary_labeled_230201_clinvar.csv",
                        help='file containing list of labeled snps')
    parser.add_argument('--file_snps_labeled_embedding', type=str, default="Binary_labeled_230201_clinvar_embedding.npy",
                        help='filescontaining embeddings of labeled snps')
    parser.add_argument('--file_snps_unlabeled', type=str, default="unlabeled_230201_clinvar.csv",
                        help='file containing list of unlabeled snps')
    parser.add_argument('--file_snps_unlabeled_embedding', type=str, default="unlabeled_230201_clinvar_embedding.npy",
                        help='file containing embeddings of unlabeled snps')
    parser.add_argument('--file_wildtype_embedding', type=str, default="wildtype_embeddings.csv",
                        help='file contaiing wildtype embeddings')
    parser.add_argument('--file_disease_embedding_LLM', type=str, default="Phenotype_embedding_CODER.csv",
                        help='file containing disease embeddings from LLM')
    parser.add_argument('--file_disease_embedding_EHR', type=str, default="Phenotype_embedding_EHRs.csv",
                        help='file containing disease embeddings from EHR')
    parser.add_argument('--file_snps_gene_map', type=str, default="Mapping_snps_genes.csv",
                        help='file containing mapping of snps to genes')
    parser.add_argument('--file_snps_prediction', type=str, default="snps_prediction.csv",
                        help='files containing list of snps for prediction')
    parser.add_argument('--file_snps_prediction_embedding', type=str, default="snps_prediction_embedding.npy",
                        help='files containing embeddings of snps for prediction')
    parser.add_argument('--flag_reload', type=int, default=0,
                        help='0-1 to denote if reload the model')
    parser.add_argument('--flag_modelsave', type=int, default=0,
                        help='0-1 to denote if save the model')
    parser.add_argument('--epochs', type=int, default=10,
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
    parser.add_argument('--flag_cross_cui', type=int, default=0,
                        help='if flag_cross_cui validation')
    parser.add_argument('--content_unlabel', type=str, default="UDN",
                        help='what unlabel SNPs to predict')
    parser.add_argument('--flag_predict', type=int, default=1,
                        help='if predict and save unlabeled snps predictions')
    parser.add_argument('--dirr_results_main', type=str, default="results/",
                        help='Directory of to save results ')
    parser.add_argument('--dirr_save_model', type=str, default="Model_save/",
                        help='Directory of to save models ')
    parser.add_argument('--dirr_pretrained_model', type=str, default="Model_pretrained/",
                        help='Directory of to pretrained models ')
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
epochs=ARGS.epochs
latent_dim=ARGS.latent_dim
batch_size=ARGS.batch_size
learning_rate=ARGS.learning_rate
weight_cosine=ARGS.weight_cosine
weight_vae= ARGS.weight_vae
weight_unlabel_snps=ARGS.weight_unlabel_snps
weight_CLIP=ARGS.weight_CLIP
train_ratio=ARGS.train_ratio
content_unlabel=ARGS.content_unlabel
flag_negative_filter=ARGS.flag_negative_filter
flag_cross_cui=ARGS.flag_cross_cui
negative_disease=ARGS.negative_disease
negative_snps=ARGS.negative_snps
margin_same=ARGS.margin_same
margin_differ=ARGS.margin_differ
weight_CLIP_snps=ARGS.weight_CLIP_snps
weight_CLIP_cui= ARGS.weight_CLIP_cui
tau_softmax=ARGS.tau_softmax
flag_hard_negative=ARGS.flag_hard_negative
flag_debug=ARGS.flag_debug
model_savename=ARGS.model_savename
flag_predict=ARGS.flag_predict
dirr_save_model= ARGS.dirr_save_model
dirr_pretrained_model=ARGS.dirr_pretrained_model
dirr_results_main=ARGS.dirr_results_main
filename_eval=ARGS.filename_eval
weight_distill=ARGS.weight_distill
margin_ppi=ARGS.margin_ppi

dirr=ARGS.main_dirr
file_CUIs_target=ARGS.file_CUIs_target
file_annotations=ARGS.file_annotations
file_snps_labeled=ARGS.file_snps_labeled
file_snps_labeled_embedding=ARGS.file_snps_labeled_embedding
file_snps_unlabeled=ARGS.file_snps_unlabeled
file_snps_unlabeled_embedding=ARGS.file_snps_unlabeled_embedding
file_wildtype_embedding=ARGS.file_wildtype_embedding
file_disease_embedding_LLM=ARGS.file_disease_embedding_LLM
file_disease_embedding_EHR=ARGS.file_disease_embedding_EHR
file_snps_gene_map=ARGS.file_snps_gene_map
 
file_snps_prediction=ARGS.file_snps_prediction
file_snps_prediction_embedding=ARGS.file_snps_prediction_embedding



scale_AE= 50.0
weight_kl=0.0
time_preprocessing=0
epoch_show = 3
if ARGS.epochs<2:
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

##################################gene embedding######
dic_gene_emb={}
df=pd.read_csv(dirr+file_wildtype_embedding)
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
df = pd.read_csv(dirr+file_annotations)
snps_labeled_all=list({}.fromkeys(list(df["snps"])).keys())
for index, snps in zip(df["snps_index"],df["snps"]):
    snps=str(snps).strip()
    if snps in dic_snps_gene:
        gene = dic_snps_gene[snps]
        dic_gene_labeled[gene] = 1
        dic_snps_index[str(snps).strip()] = int(index)
        dic_index_snps[int(index)]=str(snps).strip()
CUIs_all_covered=list(pd.read_csv(dirr+file_CUIs_target)["CUIs"])
###########snps embedding #############################snps embedding ##################

embedding=np.load(dirr+file_snps_labeled_embedding)
embedding=np.array(embedding)
df = pd.read_csv(dirr+file_snps_labeled)
SNPs=list(df[df.columns[0]])
print (" labeled SNPs: ",len(SNPs))
dic_snpsname_emb_all={}
dic_snpsname_emb_sameGENE={}
for rowi in range(len(SNPs)):
    snps_i=str(SNPs[rowi]).strip()
    if  str(SNPs[rowi]).strip() in dic_snps_gene and dic_snps_gene[str(SNPs[rowi]).strip()] in dic_gene_emb:
        gene=dic_snps_gene[snps_i]
        dic_snpsname_emb_all[snps_i] = np.array(embedding[rowi, :])
        if gene in dic_gene_labeled:
            dic_snpsname_emb_sameGENE[snps_i]=1
        if snps_i in dic_snps_index :
            dic_snps_emb[dic_snps_index[str(SNPs[rowi]).strip()]]=np.array(embedding[rowi,:])
            dic_snpsname_emb[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])

######unlabeled
embedding=np.load(dirr+file_snps_unlabeled_embedding)
embedding=np.array(embedding)
df = pd.read_csv(dirr+file_snps_unlabeled)
SNPs=list(df[df.columns[0]])
print (" unlabeled SNPs: ",len(SNPs))
for rowi in range(len(SNPs)):
    snps_i = str(SNPs[rowi]).strip()
    if str(SNPs[rowi]).strip() in dic_snps_gene and dic_snps_gene[str(SNPs[rowi]).strip()] in dic_gene_emb:
        gene = dic_snps_gene[snps_i]
        if gene in dic_gene_labeled :
            dic_snpsname_emb_sameGENE[snps_i] = 1
        dic_snpsname_emb_un[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])
        dic_snpsname_emb_all[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])

########################SNPs to be predicted##################
df_prediction=pd.read_csv(file_snps_prediction)
snps_all_prediction=list(set(list(df_prediction["SNPs"])))
for snp, gene in zip(df_prediction["SNPs"].to_list, df_prediction["Gene"].to_list):
    if gene in dic_gene_emb:
        dic_snps_gene[snp]=gene
print("SNPs to be predicted: ",len(snps_all_prediction))
embedding=np.load(dirr+file_snps_prediction_embedding)
embedding=np.array(embedding)
df = pd.read_csv(dirr+file_snps_prediction)
SNPs=list(df[df.columns[0]])
for rowi in range(len(SNPs)):
    snps_i=str(SNPs[rowi]).strip()
    if  str(SNPs[rowi]).strip() in dic_snps_gene and dic_snps_gene[str(SNPs[rowi]).strip()] in dic_gene_emb:
        gene=dic_snps_gene[snps_i]
        dic_snpsname_emb_all[snps_i] = np.array(embedding[rowi, :])
        if snps_i in dic_snps_index :
            dic_snps_emb[dic_snps_index[str(SNPs[rowi]).strip()]]=np.array(embedding[rowi,:])
            dic_snpsname_emb[str(SNPs[rowi]).strip()] = np.array(embedding[rowi, :])
########################SNPs to be predicted##################



########### readding disease embedding ##################
df_ehr=pd.read_csv(dirr+file_disease_embedding_EHR)
df_LLM=pd.read_csv(dirr+file_disease_embedding_LLM)
embedding_LLM=np.array(df_LLM[df_LLM.columns[2:]])
embedding_EHR=np.array(df_ehr[df_ehr.columns[2:]])
embedding_cui_all=[]
CUIs = list(pd.read_csv(dirr+file_disease_embedding_LLM)[df.columns[0]])
for rowi in range(len(CUIs)):
    dic_cui_emb[str(CUIs[rowi])] = np.concatenate((np.array(embedding_LLM[rowi]),np.array(embedding_EHR[rowi])),axis=-1)
    embedding_cui_all.append(np.concatenate((np.array(embedding_LLM[rowi]),np.array(embedding_EHR[rowi])),axis=-1))
print ("dic_cui_emb len: ",len(dic_cui_emb))
dic_cui_emb["benign"] = np.min(np.array(embedding_cui_all),axis=0)
########### end readding disease embedding ##################


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

def Model_PheMART(input_dim1=768,input_dim2=768+300,kl_weight=weight_kl,latent_dim=64,tau_KL=0.1):
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

    decoder1_layer1_gene = layers.Dense(int(input_dim1 * 1.5), activation=tf.nn.leaky_relu, name="decoder1/fcn1_gene",dtype='float32')
    decoder1_layer2_gene = layers.Dense(input_dim1, activation=None, name="decoder1/fcn2_gene", dtype='float32')

    decoder2_layer1_2 = layers.Dense(int(300 * 1.5), activation=tf.nn.leaky_relu, name="decoder2/fcn1_2",dtype='float32')
    decoder2_layer2_2 = layers.Dense(300, activation=None, name="decoder2/fcn2_2", dtype='float32')
    decoder2_layer1 = layers.Dense(int(768*1.5), activation=tf.nn.leaky_relu, name="decoder2/fcn1",dtype='float32')
    decoder2_layer2 = layers.Dense(768, activation=None, name="decoder2/fcn2",dtype='float32')
    # M_interaction = tf.Variable(np.random.normal(0, 1, size=(latent_dim, latent_dim)), trainable=True, dtype=tf.float32, name="interaction_M")
    def AE1(input,input_gene):
        feature = encoder1_layer1(input)
        feature_gene = encoder1_layer1(input_gene)
        feature_residual = encoder1_layer1_residual(feature-feature_gene)
        feature=feature+feature_residual
        mean=encoder1_layer2_mean(feature)
        fusioned=mean
        output=decoder1_layer1(mean)
        output=decoder1_layer2(output)
        MSE=tf.reduce_mean(tf.square(input*scale_AE-output))
        output_gene = decoder1_layer1_gene(mean)
        output_gene = decoder1_layer2_gene(output_gene)
        MSE_gene = tf.reduce_mean(tf.square(input_gene* scale_AE - output_gene))
        MSE=MSE+MSE_gene
        return fusioned, MSE,mean
    def AE2(input):
        feature_1 = encoder2_layer1_1(input[:, 0:768])
        feature_2 = encoder2_layer1_2(input[:, 768:])

        feature = tf.concat([feature_1, feature_2], axis=-1)
        mean = encoder2_layer2_mean(feature)
        output_1 = decoder2_layer1(mean)
        output_1 = decoder2_layer2(output_1)
        output_2 = decoder2_layer1_2(mean)
        output_2 = decoder2_layer2_2(output_2)
        MSE1 = tf.reduce_mean(tf.square(input[:, 0:768] * scale_AE - output_1))
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
    similarity_feature2 = tf.nn.softmax(similarity_feature2 / tau_KL, axis=1)
    similarity_input = tf.nn.softmax(similarity_input / tau_KL, axis=1)
    output = (tf.reduce_sum(tf.multiply(feature1_l, feature2_l), axis=-1)) / (
            tf.sqrt(tf.reduce_sum(tf.square(feature1_l), axis=-1)) * tf.sqrt( tf.reduce_sum(tf.square(feature2_l), axis=-1)))

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

    with tf.GradientTape(persistent=True) as tape:
        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        prediction,loss_vae,feature_snps,feature_cuis,feature1_l_gene_ppi_p,feature1_l_gene_ppi_n,feature_snp_mean,feature_snp_mean_p,feature_snp_mean_n=\
            model([input_pro_tsr, input_dis_tsr, unlabel_snps,unlabel_snps_gene,unlabel_disease,input_gene_emb,input_gene_ppi1_emb,input_gene_ppi2_emb,
                   input_pro_tsr_positive,input_pro_tsr_positive_gene,input_pro_tsr_negative,input_pro_tsr_negative_gene])

        similarity_snp_p = (tf.reduce_sum(tf.multiply(feature_snp_mean, feature_snp_mean_p), axis=-1)) / (
                    tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean), axis=-1)) * tf.sqrt( tf.reduce_sum(tf.square(feature_snp_mean_p), axis=-1)))

        similarity_snp_n = (tf.reduce_sum(tf.multiply(feature_snp_mean, feature_snp_mean_n), axis=-1)) / (tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean_n), axis=-1)))

        distance_snp_postive = tf.maximum(0, 0.2 - similarity_snp_p)
        distance_snp_negative = tf.maximum(0, similarity_snp_n - (-0.2))
        loss_snp_contrast = tf.reduce_mean(distance_snp_postive + distance_snp_negative)

        feature_snps = feature_snps / (tf.sqrt(tf.reduce_sum(tf.square(feature_snps), axis=-1, keepdims=True)))
        feature_cuis = feature_cuis / (tf.sqrt(tf.reduce_sum(tf.square(feature_cuis), axis=-1, keepdims=True)))
        feature_interaction = tf.matmul(feature_snps, feature_cuis, transpose_b=True)

        similarity_PPI_p = (tf.reduce_sum(tf.multiply(feature_snp_mean, feature1_l_gene_ppi_p), axis=-1)) / ( tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.square(feature1_l_gene_ppi_p), axis=-1)))

        similarity_PPI_n = (tf.reduce_sum(tf.multiply(feature_snp_mean, feature1_l_gene_ppi_n), axis=-1)) / (
                tf.sqrt(tf.reduce_sum(tf.square(feature_snp_mean), axis=-1)) * tf.sqrt( tf.reduce_sum(tf.square(feature1_l_gene_ppi_n), axis=-1)))

        loss_ppi = tf.maximum(similarity_PPI_n + margin_ppi - similarity_PPI_p, 0.0)*train_gene_PPI_p1_weight
        loss_ppi= tf.reduce_mean(loss_ppi)


        tau_softmax_adjust=1.0
        if tau_softmax>0.5:
            loss_snps = cce(labels_multi, tf.nn.softmax(tf.transpose(feature_interaction /tau_softmax), axis=-1))*train_gene_weight
            loss_cui = cce(labels_multi, tf.nn.softmax(feature_interaction  / tau_softmax, axis=-1))*train_gene_weight
        else:
            loss_snps = cce(labels_multi, tf.nn.softmax(tf.transpose(feature_interaction* 2.0*tau_softmax_adjust  /tau_softmax), axis=-1))*train_gene_weight
            loss_cui = cce(labels_multi, tf.nn.softmax(feature_interaction * tau_softmax_adjust / tau_softmax, axis=-1))*train_gene_weight

        loss_CLIP_P = loss_snps * label * weight_CLIP_snps + loss_cui * label * weight_CLIP_cui
        loss_CLIP = tf.reduce_mean(loss_CLIP_P )
        loss_vae = tf.reduce_mean(loss_vae)

        distance_same = tf.maximum( 0,  margin_same- prediction)
        distance_differ = tf.maximum(0, prediction - margin_differ)
        positive_ratio = (batch_size - tf.reduce_sum(label)) / (tf.reduce_sum(label) + 1)
        if weight_cosine>=5:
            loss =  tf.reduce_mean(train_gene_weight* ((batch_size - tf.reduce_sum(label))*label * distance_same*0.99 + (1 - label) * distance_differ*tf.reduce_sum(label) )/batch_size)
        else:
            loss = tf.reduce_mean(train_gene_weight*(1 - label) * distance_differ )

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
    valid_AUC.update_state(label, tf.nn.sigmoid(prediction))
    valid_PRC.update_state(label, tf.nn.sigmoid(prediction))
    return label.numpy(), prediction.numpy()

def train_model(model,ds_train, ds_test, eval_SNPs,eval_index,epochs,eval_SNPs_train,eval_cui_train,eval_gene_train,eval_gene_test):
    print("--------begin training.....")
    epoch_num = -1
    auc_total = []
    save_flag=False
    while (epoch_num < epochs):
        if epoch_num%20==5:
            MRR=[]
            rank_total=[]
            dic_snps_recall10={}
            dic_snps_recall50={}

            CUIs_all_covered=list(pd.read_csv(dirr+file_CUIs_target)["CUIs"])
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
                print("--------error in computing auROC")
                AUC_overall=0.00000000
            lr_precision, lr_recall, _ = precision_recall_curve(label_all, prediction_all)
            PRC_overall = metrics.auc(lr_recall, lr_precision)
            Prevelance_overall=1.0*np.sum(label_all)/len(label_all)
            PRC_overall_gain=1.0*(PRC_overall-Prevelance_overall)/Prevelance_overall

            print ("PRC_overall: ",PRC_overall ," prevelance: ",Prevelance_overall, "prevelance gain: ",PRC_overall_gain )
            print("---epoch: %i,  loss: %3f, VAE_loss: %3f, CLIP_loss: %3f, train_loss_ppi: %3f, train_AUC: %3f, auc_overall: %3f,AUC_SNPs: %3f, "
                  "AUC_disease: %3f prc_gain: %3f , prc_gain_snps: %3f, prc_gain_disease: %3f " % (epoch_num, train_loss.result(),VAE_loss.result(),train_loss_CLIP.result(),train_loss_ppi.result(),
                     AUC_overall_train,  np.mean(AUC_overall), np.mean(AUC_SNPs),np.mean(AUC_cuis),
                     PRC_overall_gain,np.mean(PRC_gain_SNPs),np.mean(PRC_gain_cuis)))

            if save_flag == False and epoch_num > epochs - 3:
                MRR = []
                dic_snps_recall10 = {}
                dic_snps_recall50 = {}
                rank_total=[]
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


    if flag_predict>0:
        if True:
            if True:
                if True:
                    CUIs_embedding_train = []
                    snps_embedding_train = []
                    snpsname_un=[]
                    cuis_all = list(pd.read_csv(dirr + file_CUIs_target)["CUIs"])
                    snps_all=list(snps_all_prediction)
                    gene_embedding_train=[]

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

    if flag_predict > 0:
        if True:
            if True:
                if True:

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
                    snps_num_predict = 0

                    batch_max = int(batch_size_test / batch_size)
                    for snps in snps_all_prediction:
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

                            prediction_all, CUIs_all_covered = zip(*sorted(zip(prediction_all, CUIs_all_covered), reverse=True))
                            df = pd.DataFrame({})
                            df["Phenotype"]=CUIs_all_covered
                            df["Score"]=prediction_all
                            df.to_csv(dirr_results + str(snps)+".csv",index=False)

        train_loss.reset_states()
        train_AUC.reset_states()
        VAE_loss.reset_states()
        train_loss_CLIP.reset_states()
        train_loss_ppi.reset_states()

if __name__ == '__main__':
    if not os.path.exists(dirr_save_model):
        os.makedirs(dirr_save_model)
    model = Model_PheMART(latent_dim=latent_dim)
    savename_model =  dirr_save_model+model_savename + "_model"
    if os.path.exists(dirr_pretrained_model) and flag_reload > 0:
        print("---------------------------------------------loadding saved model....................")
        model = tf.keras.models.load_model(dirr_pretrained_model)
    (traindata_snps,traindata_cuis,traindata_snps_P,traindata_cuis_P,traindata_gene_P,traindata_Y,traindata_names,
     testdata_cuis,testdata_snps,testdata_Y,testdata_names,test_pair,eval_SNPs,eval_index,
     unlabel_snps,unlabel_gene,unlabel_disease,eval_SNPs_train,eval_cui_train,train_gene,train_gene_ppi_1, train_gene_ppi_2,test_gene,eval_gene_train,eval_gene_test,train_gene_weight,
     traindata_snps_positive,traindata_snps_positive_gene,traindata_snps_negative,traindata_snps_negative_gene,train_gene_PPI_p1_weight)=\
        loaddata(train_snps_ratio=train_ratio,negative_disease=negative_disease,
                 negative_snps=negative_snps,flag_hard_mining=flag_hard_negative,flag_debug=flag_debug,
                 flag_negative_filter=flag_negative_filter,flag_cross_cui=flag_cross_cui)

    print("---loadding data end -----")
    ds_test = tf.data.Dataset.from_tensor_slices(
        (testdata_cuis, testdata_snps, testdata_Y, testdata_names, test_pair, test_gene)). \
        shuffle(buffer_size=batch_size * 1).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

    ds_train = tf.data.Dataset.from_tensor_slices((traindata_snps,traindata_cuis,traindata_Y,unlabel_snps,unlabel_gene,
                                                   unlabel_disease,train_gene,train_gene_ppi_1, train_gene_ppi_2,train_gene_weight,
                                                   traindata_snps_positive,traindata_snps_positive_gene,traindata_snps_negative,traindata_snps_negative_gene,train_gene_PPI_p1_weight)).\
        shuffle(buffer_size=batch_size*30).batch(batch_size).\
        prefetch(tf.data.experimental.AUTOTUNE).cache()

    print("---Model training begin---")
    train_model(model, ds_train,ds_test,eval_SNPs,eval_index,epochs,eval_SNPs_train,eval_cui_train,eval_gene_train,eval_gene_test)
    print("---Model training end---")
    if flag_modelsave>0:
        model.save(savename_model)
    print("---model saving end!")

