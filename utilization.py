import os.path
import numpy as np
import pandas as pd
import random
import time
from sklearn.metrics.pairwise import cosine_similarity
dirr="data/"
label_file = "Clivar_snps_disease_Pathogenic_Likely_4_11_subset.csv"    #the file with all the labeled snps
file_CUI_covered = "CUIs_all_covered_CODER_0410.csv" #the file with all the CUIs
file_label_as_pathogenic = "Clivar_snps_disease_PathogenicALL_Likely_4_11_0410.csv" 
file_PPI_labels = "PrimeKG.csv" #the file with all the PPIs
file_pathway = "PrimeKG.csv" #the file with all the pathways
file_coexpression = "gene_coexpression_gene2vec_dim_200_iter_9.txt" #the file with all the coexpression
disease_file_embedding_EHR = "Epoch_1000_L1_L2_3layer_MGB_VA_emb_embeddings_coder_clinvar_traits_cui_mapped_ALL_0410_with_all_renamed_withCUI_noOthers.csv" #the file with all the diseases's ERH embeddings
disease_file_embedding_LLM = "embeddings_coder_clinvar_traits_cui_mapped_ALL_0410_with_all_renamed_withCUI_noOthers.csv" #the file with all the diseases's LLM embeddings
snps_label_file = "ClinVar_binary_label_0201.csv"
snps_label_file_embedding = "ClinVar_binary_label_0201_embedding.npy"
snps_label_file_additional = "snps_miss_without_snp_embedding_filter.csv"
snps_label_file_additional_embedding = "ClinVar_snps_additional_embedding.npy"
snps_unlabel_file = "unlabeled_230201_clinvar.csv"
snps_unlabel_file_embedding = "unlabeled_230201_clinvar_embedding.npy"
file_emb_wild = "wildtype_embeddings.csv"
file_wildtype_embedding_unipro = "uniprotid_gene_seq_embed.csv"
file_map_snps_wild_l = "clinvar_missense.csv"
file_label_Clinvar_part = "Clinvar_missense_labeled_0410.csv"
file_map_snps_wild_u = "clinvar_missense_unlabeled.csv"
file_benign = "Clivar_snps_disease_benign_Likely_4_11_0410.csv"
file_gene_save_mapped = "Mapping_snps_genes.csv"
file_gene_save_all = "All_genes.csv"
file_hpo_embedding = "UDN_HPO_term_ALL_with_name_embedding_coder.csv"
PPI_number_threshold = 10
test_number = 10000000
###########Gene negativer protein embedding #############################snps embedding ##################


dic_gene_corexpression = {}
lines = open(dirr + file_coexpression).readlines()
print("lines", len(lines))
embedding_all = []
for line in lines:
    line = line.strip()
    gene = line.split()[0]
    vec = np.array([float(x) for x in line.split()[1:]])
    dic_gene_corexpression[gene] = vec
    embedding_all.append(vec)
print("---------------------dic_gene_corexpression", len(dic_gene_corexpression))
genes_all_with_coexpression = list(dic_gene_corexpression.keys())

##################################gene embedding######
dic_gene_emb = {}
df = pd.read_csv(dirr + file_emb_wild)
genes = list(df["gene"])
features_all = np.array(df[df.columns[1:]])
for rowi in range(len(genes)):
    gene = str(genes[rowi]).strip()
    dic_gene_emb[gene] = features_all[rowi, :]
embedding_gene_all = []
gene_list = []
for gene in dic_gene_emb:
    embedding_gene_all.append(dic_gene_emb[gene][0:768])
    gene_list.append(gene)
embedding_gene_all = np.array(embedding_gene_all)
similarity_matrix = -cosine_similarity(embedding_gene_all, embedding_gene_all)
dic_gene_negatives = {}
for rowi in range(len(gene_list)):
    gene = gene_list[rowi]
    sorted_indices = np.argsort(similarity_matrix[rowi, :])
    for index in sorted_indices:
        if not index == rowi:
            gene_negative = gene_list[index]
            dic_gene_negatives.setdefault(gene, []).append(gene_negative)

###########################################
dic_snps_gene = {}
dic_gene_snps = {}
dic_gene_label = {}
###############################################################
df = pd.read_csv(dirr + file_map_snps_wild_l)
SNPs = list(df["Name"])
genes = list(df["Gene(s)"])
for snp, gene in zip(SNPs, genes):
    splits = str(gene).split("|")
    if len(splits) > 0:
        for split in splits:
            if split in dic_gene_emb:
                dic_snps_gene[str(snp).strip()] = split
                dic_gene_snps.setdefault(split, []).append(snp)

df = pd.read_csv(dirr + file_label_Clinvar_part)
SNPs = list(df["Name"])
genes = list(df["Gene(s)"])
for snp, gene in zip(SNPs, genes):
    splits = str(gene).split("|")
    if len(splits) > 0:
        for split in splits:
            if split in dic_gene_emb:
                dic_snps_gene[str(snp).strip()] = split
                dic_gene_snps.setdefault(split, []).append(snp)

df = pd.read_csv(dirr + file_map_snps_wild_u)
SNPs = list(df["Name"])
genes = list(df["Gene(s)"])
for snp, gene in zip(SNPs, genes):
    splits = str(gene).split("|")
    if len(splits) > 0:
        for split in splits:
            if split in dic_gene_emb:
                dic_snps_gene[str(snp).strip()] = split
                dic_gene_snps.setdefault(split, []).append(str(snp).strip())
print("dic_snps_gene len: ", len(dic_snps_gene))

snps_save = []
gene_save = []
for snps in dic_snps_gene:
    snps_save.append(snps)
    gene_save.append(dic_snps_gene[snps])

df = pd.DataFrame({})
df["snps"] = snps_save
df["genes"] = gene_save
df.to_csv(dirr + file_gene_save_mapped, index=False)

df = pd.DataFrame({})
df["genes"] = list({}.fromkeys(gene_save).keys())
df.to_csv(dirr + file_gene_save_all, index=False)
#############################################################
dic_snps_cui = {}
dic_cui_snps = {}
dic_cui_emb = {}
dic_snps_emb = {}
dic_snps_index = {}
###########snps embedding #############################snps embedding ##################
embedding = np.load(dirr + snps_label_file_embedding)
embedding = np.array(embedding)
print("SNPs embedding shape: ", embedding.shape)
df = pd.read_csv(dirr + snps_label_file)
SNPs = list(df["Name"])
print(" labeled SNPs: ", len(SNPs))
dic_snps_no_gene_all = {}
for rowi in range(len(SNPs)):
    dic_snps_emb[str(SNPs[rowi])] = np.array(embedding[rowi, :])
    if str(SNPs[rowi]) in dic_snps_gene:
        if not dic_snps_gene[str(SNPs[rowi])] in dic_gene_emb:
            dic_snps_no_gene_all[str(SNPs[rowi])] = 1



for gene in dic_gene_emb:
    dic_snps_emb[gene] = dic_gene_emb[gene]
####unlabeled snps
embedding = np.load(dirr + snps_unlabel_file_embedding)
embedding = np.array(embedding)
print("unlabeled SNPs embedding shape: ", embedding.shape)
df = pd.read_csv(dirr + snps_unlabel_file)
SNPs = list(df["Name"])
print(" unlabeled SNPs: ", len(SNPs))
dic_snps_no_gene_all_unlabel = {}

for rowi in range(len(SNPs)):
    dic_snps_emb[str(SNPs[rowi])] = np.array(embedding[rowi, :])
    if str(SNPs[rowi]) in dic_snps_gene:

        if not dic_snps_gene[str(SNPs[rowi])] in dic_gene_emb:
            dic_snps_no_gene_all_unlabel[str(SNPs[rowi])] = 1

##########missed varaints
embedding = np.load(dirr + snps_label_file_additional_embedding)
embedding = np.array(embedding)
df = pd.read_csv(dirr + snps_label_file_additional)
SNPs = list(df["snp"])
dic_snps_no_gene_all_miss = {}

for rowi in range(len(SNPs)):
    dic_snps_emb[str(SNPs[rowi])] = np.array(embedding[rowi, :])
    if str(SNPs[rowi]) in dic_snps_gene:
        if not dic_snps_gene[str(SNPs[rowi])] in dic_gene_emb:
            dic_snps_no_gene_all_miss[str(SNPs[rowi])] = 1
code_save = []
for snp in dic_snps_emb:
    code_save.append(dic_snps_emb[snp])


###########snps embedding #############################snps embedding ##################
df = pd.read_csv(dirr + file_label_as_pathogenic)
snps_patho_all = set(list(df["snps"]))
dic_snps_patho = {}
for snp in snps_patho_all:
    if snp in dic_snps_emb and snp in dic_snps_gene and dic_snps_gene[snp] in dic_gene_emb:
        dic_snps_patho[snp] = 1

#####################benign SNPs#####################
dic_gene_benign = {}
df = pd.read_csv(dirr + file_benign)
SNPs = list(df["snps"])
dic_snps_benign = {}
dic_gene_snps_benign = {}
for snp in SNPs:
    snps = str(snp).strip()
    if snp in dic_snps_gene and snps in dic_snps_emb and dic_snps_gene[snp] in dic_gene_emb:
        dic_gene_benign.setdefault(dic_snps_gene[snps], []).append(snp)
        dic_snps_benign[snp] = 1
        dic_gene_snps_benign.setdefault(dic_snps_gene[snps], []).append(snp)
print("dic_gene_benign: ", len(dic_gene_benign))

df = pd.read_csv(dirr + "ClinVar_alphamissense_scores_1225_2.csv")
clinvar_name = list(df["clinvar_name"])
alphamissense_class = list(df["alphamissense_class"])
for snp, category in zip(clinvar_name, alphamissense_class):
    if snp == snp and category == category and snp in dic_snps_gene and snp in dic_snps_emb and dic_snps_gene and \
            dic_snps_gene[snp] in dic_gene_emb:
        if "benign" in category:
            dic_gene_benign.setdefault(dic_snps_gene[snp], []).append(snp)
            dic_snps_benign[snp] = 1
            dic_gene_snps_benign.setdefault(dic_snps_gene[snp], []).append(snp)
        elif "pathogenic" in category:
            if snp in dic_snps_emb and snp in dic_snps_gene and dic_snps_gene[snp] in dic_gene_emb:
                dic_snps_patho[snp] = 1
snps_patho_all = set(list(dic_snps_patho.keys()))


###########disease embedding ###########################################
df_ehr = pd.read_csv(dirr + disease_file_embedding_EHR)
df = pd.read_csv(dirr + disease_file_embedding_LLM)
embedding = np.array(df[df.columns[2:]])
embedding_cui_all = []
print("CUIs embedding shape: ", embedding.shape)
CUIs = list(pd.read_csv(dirr + disease_file_embedding_LLM)["CUIs"])
print(" labeled CUIs: ", len(CUIs))
if not len(CUIs) == len(embedding):
    print("-----------------------------------------error: ",
          "len(CUIs) unequal to len(embedding)--------------------------", "error---------------")
for rowi in range(len(CUIs)):
    cui = CUIs[rowi]
    if cui in df_ehr:
        embedding_EHR = np.array(df_ehr[cui])
        dic_cui_emb[str(CUIs[rowi])] = np.concatenate((np.array(embedding[rowi]), embedding_EHR), axis=-1)
        embedding_cui_all.append(np.concatenate((np.array(embedding[rowi]), embedding_EHR), axis=-1))
print("dic_cui_emb len: ", len(dic_cui_emb))
dic_cui_emb["benign"] = np.min(np.array(embedding_cui_all),
                               axis=0)  # np.zeros(shape=(len(embedding[0, :]),))  #   np.mean(embedding,axis=0)    #
# dic_cui_emb["others"]=  np.max(embedding,axis=0) #np.ones(shape=(len(embedding[0, :]),))*0.5   #np.sum(embedding,axis=0)  #
#


############labeled pairs######################
df = pd.read_csv(dirr + label_file)
pairs_valid = 0
print("labeled pairs reported: ", len(df["snps"]))
dic_snps_valid = {}
dic_cui_valid = {}
index_save = []
dic_non_index = {}
dic_snps_labeled = {}
dic_snps_labeled_emb = {}
dic_snps_no_gene_l = {}
dic_snp_miss = {}
dic_pair_miss = {}
dic_snp_miss_emb = {}
dic_snp_miss_gene = {}
dic_snp_miss_gene_emb = {}
dic_gene_labeled_number = {}
total_labeled_num = 0
for index, snps, cui in zip(df["snps_index"], df["snps"], df["cui"]):
    snps = str(snps).strip()
    dic_snps_index[str(snps)] = int(index)
    dic_snps_labeled[snps] = 1

    if snps in dic_snps_gene:

        if not dic_snps_gene[str(snps).strip()] in dic_gene_emb:
            dic_snps_no_gene_l[str(snps).strip()] = 1
    if snps in dic_snps_emb:
        dic_snps_labeled_emb[snps] = 1
    if snps in dic_snps_emb and cui in dic_cui_emb and snps in dic_snps_gene:
        total_labeled_num += 1
        dic_gene_label[dic_snps_gene[snps]] = 1
        dic_snps_cui.setdefault(str(snps), []).append(str(cui))
        dic_cui_snps.setdefault(str(cui), []).append(str(snps))
        pairs_valid += 1
        dic_snps_valid[snps] = 1
        dic_cui_valid[cui] = 1

        gene_l = dic_snps_gene[snps]
        if not gene_l in dic_gene_labeled_number:
            dic_gene_labeled_number[gene_l] = 1
        else:
            dic_gene_labeled_number[gene_l] += 1
    else:
        dic_snp_miss[snps] = 1
        dic_pair_miss[str(snps) + str(cui)] = 1

        if not snps in dic_snps_emb:
            dic_snp_miss_emb[snps] = 1
        if not snps in dic_snps_gene:
            dic_snp_miss_gene[snps] = 1

        if snps in dic_snps_gene and not dic_snps_gene[snps] in dic_gene_emb:
            dic_snp_miss_gene_emb[dic_snps_gene[snps]] = 1
print("----------------------dic_cui_valid : ", len(dic_cui_valid))
print("----------------------dic_pair_miss : ", len(dic_pair_miss))
print("----------------------dic_snp_miss : ", len(dic_snp_miss))
print("------------pairs_valid: ", pairs_valid)
print("dic_snps_no_gene_l len: ", len(dic_snps_no_gene_l))

print("dic_snp_miss_emb: ", len(dic_snp_miss_emb))
print("dic_snp_miss_gene: ", len(dic_snp_miss_gene))
print("dic_snp_miss_gene_emb: ", len(dic_snp_miss_gene_emb))

df = pd.DataFrame({})
df["snp"] = list(set(dic_snp_miss_emb.keys()))
df.to_csv(dirr + "snps_miss_without_snp_embedding.csv", index=False)
df = pd.DataFrame({})
df["snp"] = list(set(dic_snp_miss_gene.keys()))
df.to_csv(dirr + "snps_miss_without_mapped_genes.csv", index=False)
df = pd.DataFrame({})
df["snp"] = list(set(dic_snp_miss_gene_emb.keys()))
df.to_csv(dirr + "snps_miss_without_mapped_gene_without_embedding.csv", index=False)

################################extract snps with domains############################################
df = pd.read_csv(dirr + "domains_all_genes_new_domainONLY.csv")
genes = list(df["genes"])
starts = list(df["start"])
ends = list(df["end"])
domains = list(df["simplified"])
dic_gene_domain = {}
for gene, start, end, domain in zip(genes, starts, ends, domains):
    if ";" in domain:
        strings = domain.split(";")
        for string in strings:
            string = string.strip().lower()
            dic_gene_domain.setdefault(gene, []).append((int(start), int(end), string))
    else:
        domain = domain.strip().lower()
        dic_gene_domain.setdefault(gene, []).append((int(start), int(end), domain))
print("------------dic_gene_domain len: ", len(dic_gene_domain))


def extract_number(string):
    number = ''
    for char in string:
        if char.isdigit():
            number += char
    if number:
        return int(number)
    else:
        return -1


dic_snp_domain = {}
dic_gene_snps_domain = {}
dic_domain_snp = {}
gene_valid_snps_with_domain = {}
for snp in dic_snps_gene:
    if snp in dic_snps_emb:
        gene = dic_snps_gene[snp]
        position = str(snp).split(".")[-1]
        position = extract_number(position)
        if gene in dic_gene_domain and position > 0:
            gene_valid_snps_with_domain[gene] = 1
            for start, end, domain in dic_gene_domain[gene]:
                if start <= position <= end:
                    dic_snp_domain.setdefault(snp, []).append(domain)
                    dic_domain_snp.setdefault(domain, []).append(snp)
                    dic_gene_snps_domain.setdefault(gene, []).append(snp)
print("----dic_snp_domain len: ", len(dic_snp_domain))
print("------gene_valid_snps_with_domain len: ", len(gene_valid_snps_with_domain))
################################extract snps with domains############################################

CUIs_all_covered = list(dic_cui_valid.keys())
df = pd.DataFrame({"CUIs": CUIs_all_covered})
df.to_csv(dirr + file_CUI_covered, index=False)

snps_with_embedding = dic_snps_emb.keys()
snps_with_gene = dic_snps_gene.keys()

snps_with_embedding_gene = list(set(snps_with_embedding).intersection(set(snps_with_gene)))


dic_ppi_HiUnion = {}
dic_ENSG_gene = {}
lines = open(dirr + "ENSG_geneName_mapping.txt").readlines()
for line in lines[1:]:
    ENSG, gene = line.strip().split("\t")
    ENSG = ENSG.split(".")[0]
    dic_ENSG_gene[ENSG] = str(gene).strip()
print("dic_ENSG_gene len: ", len(dic_ENSG_gene))
df = pd.read_csv(dirr + "PPI_HI-union.csv")
genes1 = list(df["protein1"])
genes2 = list(df["protein2"])
for gene1, gene2 in zip(genes1, genes2):
    if gene1 in dic_ENSG_gene and gene2 in dic_ENSG_gene:
        dic_ppi_HiUnion.setdefault(dic_ENSG_gene[gene1], []).append(dic_ENSG_gene[gene2])
        dic_ppi_HiUnion.setdefault(dic_ENSG_gene[gene2], []).append(dic_ENSG_gene[gene1])
PPI_numbers_HiUnion = []
print("dic_ppi_HiUnion len: ", len(dic_ppi_HiUnion))
dic_ppi_HiUnion_valid = {}
dic_pair_ppi_HiUnion_valid = 0
for gene in dic_ppi_HiUnion:
    if not len(set(dic_ppi_HiUnion[gene])) > PPI_number_threshold:
        dic_ppi_HiUnion_valid[gene] = 1
        PPI_numbers_HiUnion.append(len(dic_ppi_HiUnion[gene]))

PPI_numbers_HiUnion.sort()

dic_ppi_HINT = {}
lines = open(dirr + "HomoSapiens_binary_hq.txt").readlines()
dic_unipro_gene = {}
for line in lines[1:]:
    strings = line.strip().split("\t")
    gene1 = str(strings[2]).strip()
    gene2 = str(strings[3]).strip()

    uniID1 = str(strings[0]).strip()
    uniID2 = str(strings[1]).strip()
    dic_unipro_gene[uniID1] = gene1
    dic_unipro_gene[uniID2] = gene2

    if gene1 in dic_gene_emb and gene2 in dic_gene_emb:
        dic_ppi_HINT.setdefault(gene1, []).append(gene2)
        dic_ppi_HINT.setdefault(gene2, []).append(gene1)
PPI_numbers_HINT = []

dic_ppi_HINT_valid = {}
ppi_HINT_all_valid = 0
print("dic_ppi_HINT len: ", len(dic_ppi_HINT))
for gene in dic_ppi_HINT:
    if not len(set(dic_ppi_HINT[gene])) > PPI_number_threshold:
        ppi_HINT_all_valid += len(set(dic_ppi_HINT[gene]))
        dic_ppi_HINT_valid[gene] = 1
        PPI_numbers_HINT.append(len(set(dic_ppi_HINT[gene])))
PPI_numbers_HINT.sort()

dic_ppi_HINT_HQ = {}
lines = open(dirr + "HomoSapiens_htb_hq.txt").readlines()
for line in lines[1:]:
    strings = line.strip().split("\t")
    gene1 = str(strings[2]).strip()
    gene2 = str(strings[3]).strip()
    uniID1 = str(strings[0]).strip()
    uniID2 = str(strings[1]).strip()
    dic_unipro_gene[uniID1] = gene1
    dic_unipro_gene[uniID2] = gene2
    if gene1 in dic_gene_emb and gene2 in dic_gene_emb:
        dic_ppi_HINT_HQ.setdefault(gene1, []).append(gene2)
        dic_ppi_HINT_HQ.setdefault(gene2, []).append(gene1)
PPI_numbers_HINT_HQ = []
print("dic_ppi_HINT len: ", len(dic_ppi_HINT_HQ))
dic_ppi_HINT_HQ_valid = {}
for gene in dic_ppi_HINT_HQ:
    if not len(set(dic_ppi_HINT_HQ[gene])) > PPI_number_threshold:
        dic_ppi_HINT_HQ_valid[gene] = 1
        PPI_numbers_HINT_HQ.append(len(set(dic_ppi_HINT_HQ[gene])))
PPI_numbers_HINT_HQ.sort()

dic_ppi_HINT_interface_gene1_gene2 = {}
dic_ppi_HINT_interface_gene_valid = {}
dic_ppi_HINT_interface = {}
lines = open(dirr + "H_sapiens_interfacesALL.txt").readlines()

dic_ppi_HINT_interfac_genes = {}
dic_ppi_HINT_interfac_position_pairs = {}
dic_ppi_HINT_interfac_position_gene_gene = {}

for line in lines[1:test_number]:
    strings = line.strip().split("\t")
    gene1 = str(strings[0]).strip()
    gene2 = str(strings[1]).strip()
    if gene1 in dic_unipro_gene:
        gene1 = dic_unipro_gene[gene1]
    if gene2 in dic_unipro_gene:
        gene2 = dic_unipro_gene[gene2]
    if gene1 in dic_gene_emb and gene2 in dic_gene_emb:
        pair1 = gene1 + "_" + gene2
        pair2 = gene2 + "_" + gene1
        dic_ppi_HINT_interface.setdefault(gene1, []).append(gene2)
        dic_ppi_HINT_interface.setdefault(gene2, []).append(gene1)

        postion1_string = str(strings[3]).strip()
        postion2_string = str(strings[4]).strip()
        position1 = str(postion1_string[1:-1])
        position2 = str(postion2_string[1:-1])
        if len(position1) > 0 and len(position2) > 0:
            positions1 = position1.split(",")
            positions2 = position2.split(",")

            dic_ppi_HINT_interfac_genes[pair1] = 1
            dic_ppi_HINT_interfac_position_gene_gene.setdefault(gene1, []).append(gene2)
            dic_ppi_HINT_interfac_position_gene_gene.setdefault(gene2, []).append(gene1)
            dic_ppi_HINT_interface_gene_valid[gene1] = 1
            dic_ppi_HINT_interface_gene_valid[gene2] = 1

            for pos in positions1:
                if not "-" in pos:
                    pos = int(pos)
                    for pos2 in positions2:
                        if not "-" in pos2:
                            pos2 = int(pos2)
                            dic_ppi_HINT_interface_gene1_gene2.setdefault(pair1, {}).setdefault(pos, []).append(pos2)
                            dic_ppi_HINT_interface_gene1_gene2.setdefault(pair2, {}).setdefault(pos2, []).append(pos)
                            gene_gene_pos1_pos2 = gene1 + "_" + gene2 + "_" + str(pos) + "_" + str(pos2)
                            dic_ppi_HINT_interfac_position_pairs[gene_gene_pos1_pos2] = 1

                        else:
                            pos2_start = int(pos2.split("-")[0])
                            pos2_end = int(pos2.split("-")[1])
                            if pos2_end > pos2_start:
                                for pos2_i in range(pos2_start, pos2_end + 1):
                                    dic_ppi_HINT_interface_gene1_gene2.setdefault(pair1, {}).setdefault(pos, []).append(
                                        pos2_i)
                                    dic_ppi_HINT_interface_gene1_gene2.setdefault(pair2, {}).setdefault(pos2_i,
                                                                                                        []).append(pos)
                                    gene_gene_pos1_pos2 = gene1 + "_" + gene2 + "_" + str(pos) + "_" + str(pos2_i)
                                    dic_ppi_HINT_interfac_position_pairs[gene_gene_pos1_pos2] = 1
                else:
                    pos1_start = int(pos.split("-")[0])
                    pos1_end = int(pos.split("-")[1])

                    for pos in range(pos1_start, pos1_end + 1):
                        for pos2 in positions2:
                            if not "-" in pos2:
                                pos2 = int(pos2)
                                dic_ppi_HINT_interface_gene1_gene2.setdefault(pair1, {}).setdefault(pos, []).append(
                                    pos2)
                                dic_ppi_HINT_interface_gene1_gene2.setdefault(pair2, {}).setdefault(pos2, []).append(
                                    pos)
                                gene_gene_pos1_pos2 = gene1 + "_" + gene2 + "_" + str(pos) + "_" + str(pos2)
                                dic_ppi_HINT_interfac_position_pairs[gene_gene_pos1_pos2] = 1
                            else:
                                pos2_start = int(pos2.split("-")[0])
                                pos2_end = int(pos2.split("-")[1])
                                if pos2_end > pos2_start:
                                    for pos2_i in range(pos2_start, pos2_end + 1):
                                        dic_ppi_HINT_interface_gene1_gene2.setdefault(pair1, {}).setdefault(pos,
                                                                                                            []).append(
                                            pos2_i)
                                        dic_ppi_HINT_interface_gene1_gene2.setdefault(pair2, {}).setdefault(pos2_i,
                                                                                                            []).append(
                                            pos)
                                        gene_gene_pos1_pos2 = gene1 + "_" + gene2 + "_" + str(pos) + "_" + str(pos2_i)
                                        dic_ppi_HINT_interfac_position_pairs[gene_gene_pos1_pos2] = 1

print("dic_ppi_HINT_interface_gene_valid len: ", len(dic_ppi_HINT_interface_gene_valid))
print("---------------------------dic_ppi_HINT_interface_gene1_gene2 len: ", len(dic_ppi_HINT_interface_gene1_gene2))
print("---------------------------dic_ppi_HINT_interfac_position_pairs len: ",
      len(dic_ppi_HINT_interfac_position_pairs))
print("dic_ppi_HINT_interface len: ", len(dic_ppi_HINT_interface))

PPI_numbers_HINT_interface = []
print("dic_ppi_HINT_interface len: ", len(dic_ppi_HINT_interface))
for gene in dic_ppi_HINT_interface:
    PPI_numbers_HINT_interface.append(len(set(dic_ppi_HINT_interface[gene])))
PPI_numbers_HINT_interface.sort()
print("----PPI_numbers_HINT_interface[0:50]:---- ", PPI_numbers_HINT_interface[0:50])
print("---PPI_numbers_HINT_interface[-50]:--- ", PPI_numbers_HINT_interface[-50:])
print("----PPI_numbers_HINT_interface mean: ---", np.mean(PPI_numbers_HINT_interface))
print("----PPI_numbers_HINT_interface median: ---", np.median(PPI_numbers_HINT_interface))

df = pd.read_csv(dirr + file_PPI_labels)
display_relations = list(df["display_relation"])
x_names = list(df["x_name"])
y_names = list(df["y_name"])
dic_ppi_initial = {}
dic_ppi_primkg = {}
for relation, x_name, y_name in zip(display_relations, x_names, y_names):
    if relation == "ppi" and x_name in dic_gene_emb and y_name in dic_gene_emb and x_name != y_name and x_name in dic_gene_label and y_name in dic_gene_label:
        dic_ppi_primkg.setdefault(x_name, []).append(y_name)
        dic_ppi_primkg.setdefault(y_name, []).append(x_name)
for gene in dic_ppi_initial:
    dic_ppi_primkg[gene] = list(set(dic_ppi_initial[gene]))

dic_gene_label_ppi = {}
genes_all_with_ppi = list(dic_ppi_primkg.keys())
ppi_length_label = []

df = pd.read_csv(dirr + file_pathway)
relations = list(df["relation"])
x_types = list(df["x_type"])
x_names = list(df["x_name"])
y_types = list(df["y_type"])
y_names = list(df["y_name"])
dic_gene_process = {}
for relation, x_type, x_name, y_type, y_name in zip(relations, x_types, x_names, y_types, y_names):
    if relation == "bioprocess_protein" and x_type == "gene/protein" and y_type == "biological_process" and x_name in dic_gene_emb and x_name in dic_gene_label:
        dic_gene_process.setdefault(x_name, []).append(y_name)

dic_process_gene_gene_inital = {}
for gene1 in dic_gene_process:
    for gene2 in dic_gene_process:
        if gene1 != gene2:
            process1 = dic_gene_process[gene1]
            process2 = dic_gene_process[gene2]
            if len(set(process1).intersection(set(process2))) > 0:
                dic_process_gene_gene_inital.setdefault(gene1, []).append(gene2)
                dic_process_gene_gene_inital.setdefault(gene2, []).append(gene1)

dic_process_gene_gene = {}
for gene in dic_process_gene_gene_inital:
    dic_process_gene_gene[gene] = list(set(dic_process_gene_gene_inital[gene]))

dic_ppi_pathway = {}
genes_pathway_ppi = list(set(dic_ppi_primkg.keys()).union(set(dic_process_gene_gene.keys())))
for gene in genes_pathway_ppi:
    if gene in dic_process_gene_gene and gene in dic_ppi_primkg:
        dic_ppi_pathway[gene] = list(set(dic_ppi_primkg[gene]).union(set(dic_process_gene_gene[gene])))
    else:
        if gene in dic_ppi_primkg:
            dic_ppi_pathway[gene] = dic_ppi_primkg[gene]
        else:
            dic_ppi_pathway[gene] = dic_process_gene_gene[gene]
print("dic_ppi_pathway_union len: ", len(dic_ppi_pathway))

########################################getting the weights of genes based on the annotated paris###############
print("dic_gene_labeled_number: ", len(dic_gene_labeled_number), "total_labeled_num: ", total_labeled_num)
dic_gene_weight = {}
gene_weights_all = []
for gene in dic_gene_labeled_number:
    dic_gene_weight[gene] = 1 / np.log2(1 + 3000.0 * dic_gene_labeled_number[gene] / total_labeled_num)
    gene_weights_all.append(1 / np.log2(1 + 3000.0 * dic_gene_labeled_number[gene] / total_labeled_num))
gene_weghts_median = np.median(gene_weights_all)
gene_weights_all.sort()
########################################getting the weights of genes based on the annotated paris###############


snps_total = list(set(list(dic_snps_cui.keys())))
snps_total = list(snps_total)
random.shuffle(snps_total)


def extract_number(string):
    number = ''
    for char in string:
        if char.isdigit():
            number += char
    if number:
        return int(number)
    else:
        return -1


########################################geting the snps postives/negatives based on PPIs###############
genes_labeled_all = list(dic_gene_label.keys())
dic_snps_PPI_interface_positives = {}
dic_snps_PPI_HINT_HQ_positives = {}
dic_snps_PPI_HINT_positives = {}
dic_snps_PPI_HIU_positives = {}
dic_snps_PPI_domain_positives = {}
dic_snps_PPI_interface_negatives = {}
dic_snps_PPI_HINT_HQ_negatives = {}
dic_snps_PPI_HINT_negatives = {}
dic_snps_PPI_HIU_negatives = {}
dic_snps_PPI_domain_negatives = {}
dic_position_positive_hit_gene = {}
dic_position_positive_hit_variant_pair = {}
dic_position_positive_hit_gene_pair = {}
flag_variant_test = 0
print("---dic_snps_gene.keys() len: ", len(dic_snps_gene.keys()))
dic_snp_ppi_domain_valid_all = {}
snps_all_benign = list(dic_snps_benign.keys())
for snp in list(dic_snps_patho.keys())[0:test_number]:
    flag_variant_test += 1
    gene = dic_snps_gene[snp]
    position = str(snp).split(".")[-1]
    position = extract_number(position)
    if flag_variant_test % 50000 == 4999:
        print("flag_variant_test: ", flag_variant_test)
        print("---------------------------------------dic_position_positive_hit_variant_pair len: ",
              len(dic_position_positive_hit_variant_pair))
        print("---------------------------------------dic_position_positive_hit_gene len: ",
              len(dic_position_positive_hit_gene))
        print("---------------------------------------dic_position_positive_hit_gene_pair len: ",
              len(dic_position_positive_hit_gene_pair))
    if position > 0 and gene in dic_ppi_HINT_interfac_position_gene_gene:
        genes2 = set(dic_ppi_HINT_interfac_position_gene_gene[gene])
        for gene2 in genes2:
            pair_key = gene + "_" + gene2
            if pair_key in dic_ppi_HINT_interface_gene1_gene2 and position in dic_ppi_HINT_interface_gene1_gene2[
                pair_key]:
                snps_gene2 = dic_gene_snps[gene2]
                for snp_p in snps_gene2:
                    position_p = str(snp_p).split(".")[-1]
                    position_p = extract_number(position_p)
                    if snp_p in snps_patho_all and position_p in dic_ppi_HINT_interface_gene1_gene2[pair_key][position]:
                        dic_snps_PPI_interface_positives.setdefault(snp, []).append(snp_p)
                        dic_position_positive_hit_variant_pair[snp + snp_p] = 1
                        dic_position_positive_hit_gene[gene] = 1
                        dic_position_positive_hit_gene_pair[gene + "_" + gene2] = 1
                        dic_position_positive_hit_gene_pair[gene2 + "_" + gene] = 1
                        dic_snp_ppi_domain_valid_all[snp] = 1

        genes_negatives = list(set(genes_labeled_all) - set(dic_ppi_HINT_interfac_position_gene_gene[gene]))
        if gene in dic_ppi_HiUnion:
            genes_negatives = list(set(genes_negatives) - set(dic_ppi_HiUnion[gene]))
        if gene in dic_ppi_pathway:
            genes_negatives = list(set(genes_negatives) - set(dic_ppi_pathway[gene]))
        number_valid = 0
        for gene_n in dic_gene_negatives[gene][0:50]:
            if gene_n in genes_negatives and number_valid < 5:
                if gene_n != gene:
                    snps_temp = dic_gene_snps[gene_n]
                    for snp_ii in random.choices(snps_temp, k=5):
                        if snp_ii in dic_snps_emb:
                            number_valid += 1
                            dic_snps_PPI_interface_negatives.setdefault(snp, []).append(snp_ii)
        if snp in dic_snps_PPI_interface_positives and not snp in dic_snps_PPI_interface_negatives:
            dic_snps_PPI_interface_negatives.setdefault(snp, []).append(random.choice(snps_all_benign))

    if not snp in dic_snp_ppi_domain_valid_all and gene in dic_ppi_HINT_HQ:
        dic_gene_temp = dic_ppi_HINT_HQ
        if len(dic_gene_temp) > 0:
            if gene in dic_gene_temp:
                for gene_p in dic_gene_temp[gene]:
                    if gene_p != gene and gene_p in dic_gene_label and gene_p in dic_gene_emb:
                        snps_temp = dic_gene_snps[gene_p]
                        for snp_ii in random.choices(snps_temp, k=20):
                            if snp_ii in snps_patho_all:
                                dic_snps_PPI_HINT_HQ_positives.setdefault(snp, []).append(snp_ii)
                                dic_snp_ppi_domain_valid_all[snp] = 1
                genes_negatives = list(set(genes_labeled_all) - set(dic_gene_temp[gene]))
                if gene in dic_ppi_HiUnion:
                    genes_negatives = list(set(genes_negatives) - set(dic_ppi_HiUnion[gene]))
                if gene in dic_ppi_pathway:
                    genes_negatives = list(set(genes_negatives) - set(dic_ppi_pathway[gene]))
                number_valid = 0
                for gene_n in dic_gene_negatives[gene][0:50]:
                    if gene_n != gene and gene_n in dic_gene_emb and number_valid < 5:
                        snps_temp = dic_gene_snps[gene_n]
                        for snp_ii in random.choices(snps_temp, k=4):
                            if snp_ii in dic_snps_emb:
                                number_valid += 1
                                dic_snps_PPI_HINT_HQ_negatives.setdefault(snp, []).append(snp_ii)

                if snp in dic_snps_PPI_HINT_HQ_positives and not snp in dic_snps_PPI_HINT_HQ_negatives:
                    dic_snps_PPI_HINT_HQ_negatives.setdefault(snp, []).append(random.choice(snps_all_benign))

    if not snp in dic_snp_ppi_domain_valid_all and gene in dic_ppi_HINT:
        dic_gene_temp = dic_ppi_HINT
        if len(dic_gene_temp) > 0:
            if gene in dic_gene_temp:
                for gene_p in dic_gene_temp[gene]:
                    if gene_p != gene and gene_p in dic_gene_label and gene_p in dic_gene_emb:
                        snps_temp = dic_gene_snps[gene_p]
                        for snp_ii in random.choices(snps_temp, k=20):
                            if snp_ii in snps_patho_all:
                                dic_snps_PPI_HINT_positives.setdefault(snp, []).append(snp_ii)
                                dic_snp_ppi_domain_valid_all[snp] = 1
                genes_negatives = list(set(genes_labeled_all) - set(dic_gene_temp[gene]))
                if gene in dic_ppi_HiUnion:
                    genes_negatives = list(set(genes_negatives) - set(dic_ppi_HiUnion[gene]))
                if gene in dic_ppi_pathway:
                    genes_negatives = list(set(genes_negatives) - set(dic_ppi_pathway[gene]))
                number_valid = 0
                for gene_n in dic_gene_negatives[gene][0:50]:
                    if gene_n != gene and gene_n in dic_gene_emb and number_valid < 5:
                        snps_temp = dic_gene_snps[gene_n]
                        for snp_ii in random.choices(snps_temp, k=3):
                            if snp_ii in dic_snps_emb:
                                number_valid += 1
                                dic_snps_PPI_HINT_negatives.setdefault(snp, []).append(snp_ii)

                if snp in dic_snps_PPI_HINT_positives and not snp in dic_snps_PPI_HINT_negatives:
                    dic_snps_PPI_HINT_negatives.setdefault(snp, []).append(random.choice(snps_all_benign))

    if not snp in dic_snp_ppi_domain_valid_all and gene in dic_ppi_HiUnion:
        dic_gene_temp = dic_ppi_HiUnion
        if len(dic_gene_temp) > 0:
            if gene in dic_gene_temp:
                for gene_p in dic_gene_temp[gene]:
                    if gene_p != gene and gene_p in dic_gene_emb:
                        snps_temp = dic_gene_snps[gene_p]
                        for snp_ii in random.choices(snps_temp, k=8):
                            if snp_ii in snps_patho_all:
                                dic_snps_PPI_HIU_positives.setdefault(snp, []).append(snp_ii)
                                dic_snp_ppi_domain_valid_all[snp] = 1
                genes_negatives = list(set(genes_labeled_all) - set(dic_gene_temp[gene]))
                if gene in dic_ppi_HiUnion:
                    genes_negatives = list(set(genes_negatives) - set(dic_ppi_HiUnion[gene]))
                if gene in dic_ppi_pathway:
                    genes_negatives = list(set(genes_negatives) - set(dic_ppi_pathway[gene]))
                number_valid = 0
                for gene_n in dic_gene_negatives[gene][0:50]:
                    if gene_n != gene and gene_n in dic_gene_emb and number_valid < 5:
                        snps_temp = dic_gene_snps[gene_n]
                        for snp_ii in random.choices(snps_temp, k=2):
                            if snp_ii in dic_snps_emb:
                                number_valid += 1
                                dic_snps_PPI_HIU_negatives.setdefault(snp, []).append(snp_ii)

                if snp in dic_snps_PPI_HIU_positives and not snp in dic_snps_PPI_HIU_negatives:
                    dic_snps_PPI_HIU_negatives.setdefault(snp, []).append(random.choice(snps_all_benign))

print("---------------------------------------dic_position_positive_hit_variant_pair len: ",
      len(dic_position_positive_hit_variant_pair))
print("---------------------------------------dic_position_positive_hit_gene len: ",
      len(dic_position_positive_hit_gene))
print("---------------------------------------dic_position_positive_hit_gene_pair len: ",
      len(dic_position_positive_hit_gene_pair))
print("dic_snps_PPI_interface_positives len: ", len(dic_snps_PPI_interface_positives))
print("dic_snps_PPI_interface_negatives len: ", len(dic_snps_PPI_interface_negatives))
print("dic_snps_PPI_HINT_HQ_positives len: ", len(dic_snps_PPI_HINT_HQ_positives))
print("dic_snps_PPI_HINT_HQ_negatives len: ", len(dic_snps_PPI_HINT_HQ_negatives))
print("dic_snps_PPI_HINT_positives len: ", len(dic_snps_PPI_HINT_positives))
print("dic_snps_PPI_HINT_negatives len: ", len(dic_snps_PPI_HINT_negatives))
print("dic_snps_PPI_HIU_positives len: ", len(dic_snps_PPI_HIU_positives))
print("dic_snps_PPI_HIU_negatives len: ", len(dic_snps_PPI_HIU_negatives))

########################################geting the snps postives/negatives based on PPIs###############


########################################geting the domain information of SNPs###############
genes_labeled_all = list(dic_gene_label.keys())
dic_snps_domain_negatives = {}
dic_snps_domain_positives = {}
snps_all_with_domain = list(set(dic_snp_domain.keys()))
print("snps_all_with_domain: ", len(snps_all_with_domain))
for snp in snps_all_with_domain[0:test_number]:
    if snp in dic_snps_gene and snp in snps_patho_all:
        position = str(snp).split(".")[-1]
        position = extract_number(position)
        gene = dic_snps_gene[snp]
        for domain in set(dic_snp_domain[snp]):
            if domain in dic_domain_snp:
                for snp_p in set(dic_domain_snp[domain]):
                    if snp_p in snps_patho_all:
                        dic_snps_domain_positives.setdefault(snp, []).append(snp_p)
                        dic_snp_ppi_domain_valid_all[snp] = 1

                snps_temp = dic_gene_snps_domain[gene]
                for snp_ii in snps_temp:
                    position_n = str(snp_ii).split(".")[-1]
                    position_n = extract_number(position_n)
                    if not domain in dic_snp_domain[snp_ii] and abs(position_n - position) > 200:
                        dic_snps_domain_negatives.setdefault(snp, []).append(snp_ii)

                number_valid = 0
                for gene_n in dic_gene_negatives[gene][0:100]:
                    if number_valid < 5 and gene_n != gene and gene_n in dic_gene_emb and gene_n in dic_gene_snps_domain:
                        snps_temp = dic_gene_snps_domain[gene_n]
                        for snp_ii in random.choices(snps_temp, k=5):
                            if not domain in dic_snp_domain[snp_ii]:
                                number_valid += 1
                                dic_snps_domain_negatives.setdefault(snp, []).append(snp_ii)
            else:
                for snp_p in dic_gene_snps[gene][0:400]:
                    if not snp_p == snp and snp_p in snps_patho_all:
                        position_p = str(snp_p).split(".")[-1]
                        position_p = extract_number(position_p)
                        if abs(position_p - position) < 12:
                            dic_snps_domain_positives.setdefault(snp, []).append(snp_p)

        if snp in dic_snps_domain_positives and not snp in dic_snps_domain_negatives:
            for snp_n in random.choices(snps_all_benign, k=5):
                dic_snps_domain_negatives.setdefault(snp, []).append(snp_n)

print("--------------------------------------------------dic_snp_ppi_domain_valid_all len: ",
      len(dic_snp_ppi_domain_valid_all), "-----------------------------------")
print("---------------------------------------dic_position_positive_hit_variant_pair len: ",
      len(dic_position_positive_hit_variant_pair))
print("---------------------------------------dic_position_positive_hit_gene len: ",
      len(dic_position_positive_hit_gene))
print("---------------------------------------dic_position_positive_hit_gene_pair len: ",
      len(dic_position_positive_hit_gene_pair))
print("dic_snps_domain_positives len: ", len(dic_snps_domain_positives))
print("dic_snps_domain_negatives len: ", len(dic_snps_domain_negatives))


########################################geting the domain information of SNPs###############


def cosine_similarity_my(v1, v2):
    # Compute the dot product of vectors v1 and v2
    dot_product = np.dot(v1, v2)
    # Compute the norm (magnitude) of each vector
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # Compute cosine similarity
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity


dic_cui_cui_similarities = {}
for cui in dic_cui_snps:
    for cui2 in dic_cui_snps:
        dic_cui_cui_similarities[(cui, cui2)] = cosine_similarity_my(dic_cui_emb[cui], dic_cui_emb[cui2])
        dic_cui_cui_similarities[(cui2, cui)] = dic_cui_cui_similarities[(cui, cui2)]

dic_snp_positive_snps = {}
dic_snp_negative_snps = {}
snps_labeled_all = list(dic_snps_cui.keys())
for snp in snps_labeled_all[0:test_number]:
    position = str(snp).split(".")[-1]
    position = extract_number(position)

    cui_embedding_label = []
    for cui_l in dic_snps_cui[snp]:
        cui_embedding_label.append(dic_cui_emb[cui_l])
        for cui_p in dic_cui_snps:
            if dic_cui_cui_similarities[(cui_l, cui_p)] > 0.8 and cui_p != cui_l and cui_p in dic_cui_emb:
                for snp_p in dic_cui_snps[cui_p]:
                    if snp_p in dic_snps_emb and snp_p != snp and snp_p in dic_snps_gene:
                        dic_snp_positive_snps.setdefault(snp, []).append(snp_p)
    # if not snp in dic_snp_positive_snps:
    #     dic_snp_positive_snps.setdefault(snp, []).append(snp)
    cui_embedding_label = np.array(cui_embedding_label)
    gene = dic_snps_gene[snp]
    for snp_n in dic_gene_snps[gene]:
        position_n = str(snp_n).split(".")[-1]
        position_n = extract_number(position_n)
        if snp_n in dic_snps_cui and snp_n in dic_snps_emb and snp_n != snp and abs(position_n - position) > 160:
            cui_embedding_n = []
            for cui_i in dic_snps_cui[snp_n]:
                cui_embedding_n.append(dic_cui_emb[cui_i])
            cui_embedding_n = np.array(cui_embedding_n)
            similarity_n_l = cosine_similarity(cui_embedding_label, cui_embedding_n)
            if np.max(similarity_n_l) < 0.6:
                dic_snp_negative_snps.setdefault(snp, []).append(snp_n)
    gene_valid_num = 10
    if gene in dic_gene_negatives:
        for gene_n in dic_gene_negatives[gene][0:50]:
            if gene_n in dic_gene_snps_benign and gene_valid_num > 0 and gene_n in dic_gene_emb:
                gene_valid_num -= 1
                snps_temp = dic_gene_snps_benign[gene_n]
                for snp_n in random.choices(snps_temp, k=2):
                    if snp_n in dic_snps_emb:
                        dic_snp_negative_snps.setdefault(snp, []).append(snp_n)
    # if not snp in dic_snp_negative_snps:
    #     dic_snp_negative_snps.setdefault(snp, []).append(random.choice(snps_labeled_all))
    # dic_snp_negative_snps.setdefault(snp, []).append(gene)
print("---dic_snp_positive_snps len---: ", len(dic_snp_positive_snps))
print("---dic_snp_negative_snps len---: ", len(dic_snp_negative_snps))
snps_all_benign = list(dic_snps_benign.keys())

if True:
    for snp in dic_snps_gene:
        gene = dic_snps_gene[snp]
        if gene in dic_gene_emb:
            if not snp in dic_snp_positive_snps:  #############if not positive SNPs included, include the neighboring SNPs as the postives based on the contiguity of functional domains
                position = str(snp).split(".")[-1]
                position = extract_number(position)
                for snp_p in dic_gene_snps[gene][0:300]:
                    if not snp_p == snp and snp_p in dic_snps_emb and snp_p in dic_snps_gene and snp_p in snps_patho_all:
                        position_p = str(snp_p).split(".")[-1]
                        position_p = extract_number(position_p)
                        if abs(position_p - position) < 12:
                            dic_snp_positive_snps.setdefault(snp, []).append(snp_p)

            if not snp in dic_snp_positive_snps:  #####if no postives, then include itself as the positive
                dic_snp_positive_snps.setdefault(snp, []).append(snp)

            if not snp in dic_snp_negative_snps:  #####if no negatives, then include the benign snps on the genes with higher similarity to its gene
                gene_valid_num = 10
                if gene in dic_gene_negatives:
                    for gene_n in dic_gene_negatives[gene][0:50]:
                        if gene_n in dic_gene_snps_benign and gene_valid_num > 0 and gene_n in dic_gene_emb:
                            gene_valid_num -= 1
                            snps_temp = dic_gene_snps_benign[gene_n]
                            for snp_n in random.choices(snps_temp, k=2):
                                if snp_n in dic_snps_emb:
                                    dic_snp_negative_snps.setdefault(snp, []).append(snp_n)

                while (not snp in dic_snp_negative_snps) or len(dic_snp_negative_snps[
                                                                    snp]) < gene_valid_num:  #####if no negatives, then include any benign snps
                    snp_n = random.choice(snps_all_benign)
                    if snp_n in dic_snps_gene and dic_snps_gene[snp_n] in dic_gene_emb:
                        dic_snp_negative_snps.setdefault(snp, []).append(snp_n)

print("---dic_snp_positive_snps final len---: ", len(dic_snp_positive_snps))
print("---dic_snp_negative_snps final len---: ", len(dic_snp_negative_snps))


############################################getting training data#######################################################
def loaddata(train_snps_ratio=0.9, negative_disease=10, negative_snps=10, flag_hard_mining=1, negative_num_max=6,
             snps_remove=["test"],
             flag_debug=0, flag_negative_filter=1, similarith_N_threshold_max=0.75, similarith_N_threshold_min=-5.1,
             flag_cross_gene=0, flag_cross_cui=0):
    print("loaddata beginning....")

    weight_ppi_interface = 4.0
    weight_ppi_hint_hq = 0.2
    weight_ppi_hint = 0.15
    weight_ppi_hiu = 0.1
    weight_domain = 2.0

    snps_total = list(set(list(dic_snps_cui.keys())))
    snps_total = list(snps_total)
    random.shuffle(snps_total)
    if flag_cross_gene > 0:
        genes_total_label = list(dic_gene_label.keys())
        random.shuffle(genes_total_label)
        gene_train = genes_total_label[0:int(len(genes_total_label) * train_snps_ratio)]
        gene_test = genes_total_label[int(len(genes_total_label) * train_snps_ratio):]
        snps_train = []
        snps_test = []
        for snp_i in snps_total:
            gene_i = dic_snps_gene[snp_i]
            if gene_i in gene_train:
                snps_train.append(snp_i)
            else:
                snps_test.append(snp_i)
    else:
        snps_train = snps_total[0:int(len(snps_total) * train_snps_ratio)]
        snps_test = snps_total[int(len(snps_total) * train_snps_ratio):]

    cuis_total_label = list(dic_cui_valid.keys())
    if flag_cross_cui > 0:
        random.shuffle(cuis_total_label)
        CUI_train = set(cuis_total_label[0:int(len(cuis_total_label) * train_snps_ratio)])
        CUI_test = set(cuis_total_label[int(len(cuis_total_label) * train_snps_ratio):])
        snps_train = snps_total
        snps_test = snps_test
    else:
        CUI_train = cuis_total_label
        CUI_test = cuis_total_label

    snps_benign_all_train_test = list(dic_snps_benign.keys())
    random.shuffle(snps_benign_all_train_test)
    snps_benign_train = snps_benign_all_train_test[0:int(len(snps_benign_all_train_test) * train_snps_ratio)]
    snps_benign_test = snps_benign_all_train_test[int(len(snps_benign_all_train_test) * train_snps_ratio):]

    number_hard_negative = 0
    number_hard_negative_unlabel = 0
    number_hard_negative_gene = 0
    dic_gene_hit_as_unlabel = {}
    dic_gene_hit_as_benign = {}
    dic_gene_train = {}
    print('loaddata begin...')
    traindata_cuis = []
    traindata_snps = []
    traindata_snps_positive = []
    traindata_snps_positive_gene = []
    traindata_snps_negative = []
    traindata_snps_negative_gene = []

    traindata_cuis_P = []
    traindata_snps_P = []
    traindata_gene_P = []

    train_gene = []
    train_gene_weight = []
    train_gene_PPI_p1 = []
    train_gene_PPI_p1_gene = []
    train_gene_PPI_p1_weight = []
    train_gene_PPI_p2 = []
    train_gene_PPI_p2_gene = []
    traindata_Y = []
    train_pair = []
    traindata_names = []
    testdata_cuis = []
    testdata_snps = []
    testdata_gene = []
    testdata_Y = []
    testdata_names = []
    test_pair = []
    unlabel_snps = []
    unlabel_gene = []
    unlabel_disease = []
    snps_total = list(set(list(dic_snps_cui.keys())))
    snps_total = list(snps_total)
    random.shuffle(snps_total)
    dic_snps_unlabeled = {}
    dic_snps_unlabeled_gene = {}

    if flag_debug > 0:
        snps_total = list(snps_total)
        random.shuffle(snps_total)
        snps_total = snps_total[0:1000]
        negative_disease = 1
        negative_snps = 1

    eval_SNPs = []
    eval_index = []
    eval_SNPs_train = []
    eval_cui_train = []
    eval_gene_train = []
    cuis_all = list(dic_cui_emb.keys())
    snps_all = list(dic_snps_emb.keys())
    eval_gene_test = []

    dic_gene_train = {}
    dic_gene_train_in_benigh = {}

    snps_number = 0
    for snps in snps_total:

        cui_labeled = set(dic_snps_cui[snps])
        cui_unlabeled = list(set(CUIs_all_covered) - set(cui_labeled))

        snps_number += 1
        # if snps_number%5==0:
        #     print("snps_number:  ",snps_number, "len cui_unlabeled: ",len(cui_unlabeled))

        cui_label_embedding = []
        cui_unlabel_embedding = []
        for cui_l in cui_labeled:
            cui_label_embedding.append(dic_cui_emb[cui_l])
        for cui_un in cui_unlabeled:
            cui_unlabel_embedding.append(dic_cui_emb[cui_un])
        if flag_negative_filter > 0:
            cui_unlabeled_filtered = []
            cui_label_embedding = np.array(cui_label_embedding)
            cui_unlabel_embedding = np.array(cui_unlabel_embedding)
            # print("cui_label_embedding shape: ",cui_label_embedding.shape)
            # print("cui_unlabel_embedding shape: ", cui_unlabel_embedding.shape)
            similarity_un_l = cosine_similarity(cui_unlabel_embedding, cui_label_embedding)
            # print ("similarity_un_l shape: ",similarity_un_l.shape)
            # print("similarity_un_l min: ", np.min(similarity_un_l))
            # print("similarity_un_l max: ", np.max(similarity_un_l))
            # print("cui_unlabeled len: ",len(cui_unlabeled))
            for rowi in range(len(cui_unlabeled)):
                if np.max(similarity_un_l[rowi, :]) < similarith_N_threshold_max and np.max(
                        similarity_un_l[rowi, :]) > similarith_N_threshold_min \
                        and not cui_unlabeled[rowi] == "others":
                    cui_unlabeled_filtered.append(cui_unlabeled[rowi])
            # print("cui_unlabeled_filtered len: ", len(cui_unlabeled_filtered))
            cui_unlabeled = cui_unlabeled_filtered

        if snps in snps_train:  ###training SNPs
            if flag_cross_cui > 0:
                cui_labeled = set(dic_snps_cui[snps]) - CUI_test
                cui_unlabeled = list(set(CUIs_all_covered) - set(cui_labeled) - set(CUI_test))
            else:
                cui_labeled = set(dic_snps_cui[snps])
                cui_unlabeled = list(set(CUIs_all_covered) - set(cui_labeled))

            if len(cui_labeled) > 0:
                for cui in cui_labeled:
                    eval_SNPs_train.append(int(dic_snps_index[snps]))
                    eval_cui_train.append(cui)
                    gene = dic_snps_gene[snps]
                    dic_gene_train[gene] = 1
                    eval_gene_train.append(dic_gene_emb[gene])

                    traindata_cuis_P.append(dic_cui_emb[cui])
                    traindata_snps_P.append(dic_snps_emb[snps])
                    traindata_gene_P.append(dic_gene_emb[gene])

                    traindata_cuis.append(dic_cui_emb[cui])
                    traindata_snps.append(dic_snps_emb[snps])
                    snps_p_temp = random.choice(dic_snp_positive_snps[snps])
                    traindata_snps_positive.append(dic_snps_emb[snps_p_temp])
                    traindata_snps_positive_gene.append(dic_gene_emb[dic_snps_gene[snps_p_temp]])
                    snps_N_temp = random.choice(dic_snp_negative_snps[snps])
                    traindata_snps_negative.append(dic_snps_emb[snps_N_temp])
                    traindata_snps_negative_gene.append(dic_gene_emb[dic_snps_gene[snps_N_temp]])

                    gene = dic_snps_gene[snps]
                    train_gene.append(dic_gene_emb[gene])
                    train_gene_weight.append(dic_gene_weight[gene])
                    traindata_Y.append(1.0)

                    if snps in dic_snps_PPI_interface_positives:
                        snps_temp = random.choice(dic_snps_PPI_interface_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_interface_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_interface)
                    elif snps in dic_snps_domain_positives:
                        snps_temp = random.choice(dic_snps_domain_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_domain_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_domain)
                    elif snps in dic_snps_PPI_HINT_HQ_positives:
                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hint_hq)
                    elif snps in dic_snps_PPI_HINT_positives:
                        snps_temp = random.choice(dic_snps_PPI_HINT_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HINT_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hint)
                    elif snps in dic_snps_PPI_HIU_positives:
                        snps_temp = random.choice(dic_snps_PPI_HIU_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HIU_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hiu)
                    else:
                        train_gene_PPI_p1.append(dic_snps_emb[snps])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps]])
                        train_gene_PPI_p2.append(dic_gene_emb[gene])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[gene])
                        train_gene_PPI_p1_weight.append(1.0)

                    traindata_Y.append(1.0)
                    traindata_names.append(int(dic_snps_index[snps]))
                    snps_unlabel_i = snps_with_embedding_gene[random.randint(0, len(snps_with_embedding_gene) - 1)]
                    gene_unlabel_i = dic_snps_gene[snps_unlabel_i]
                    unlabel_gene.append(dic_gene_emb[gene_unlabel_i])
                    unlabel_snps.append(dic_snps_emb[snps_unlabel_i])
                    unlabel_disease.append(dic_cui_emb[cuis_all[random.randint(0, len(cuis_all) - 1)]])

                    dic_snps_unlabeled[snps_unlabel_i] = 1
                    dic_snps_unlabeled_gene[gene_unlabel_i] = 1

                    index_random = random.randint(0, len(traindata_cuis_P) - 1)
                    traindata_cuis_P.append(traindata_cuis_P[index_random])
                    traindata_snps_P.append(traindata_snps_P[index_random])

                    traindata_gene_P.append(traindata_gene_P[index_random])

                    traindata_cuis.append(dic_cui_emb["benign"])
                    traindata_snps.append(dic_snps_emb[snps])

                    snps_p_temp = random.choice(dic_snp_positive_snps[snps])
                    traindata_snps_positive.append(dic_snps_emb[snps_p_temp])
                    traindata_snps_positive_gene.append(dic_gene_emb[dic_snps_gene[snps_p_temp]])
                    snps_N_temp = random.choice(dic_snp_negative_snps[snps])
                    traindata_snps_negative.append(dic_snps_emb[snps_N_temp])
                    traindata_snps_negative_gene.append(dic_gene_emb[dic_snps_gene[snps_N_temp]])

                    gene = dic_snps_gene[snps]
                    train_gene.append(dic_gene_emb[gene])
                    train_gene_weight.append(dic_gene_weight[gene])

                    if snps in dic_snps_PPI_interface_positives:
                        snps_temp = random.choice(dic_snps_PPI_interface_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_interface_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_interface)
                    elif snps in dic_snps_domain_positives:
                        snps_temp = random.choice(dic_snps_domain_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_domain_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_domain)
                    elif snps in dic_snps_PPI_HINT_HQ_positives:
                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hint_hq)
                    elif snps in dic_snps_PPI_HINT_positives:
                        snps_temp = random.choice(dic_snps_PPI_HINT_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HINT_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hint)
                    elif snps in dic_snps_PPI_HIU_positives:
                        snps_temp = random.choice(dic_snps_PPI_HIU_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HIU_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hiu)
                    else:
                        train_gene_PPI_p1.append(dic_snps_emb[snps])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps]])
                        train_gene_PPI_p2.append(dic_gene_emb[gene])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[gene])
                        train_gene_PPI_p1_weight.append(1.0)

                    traindata_Y.append(0.0)
                    traindata_names.append(int(dic_snps_index[snps]))
                    snps_unlabel_i = snps_with_embedding_gene[random.randint(0, len(snps_with_embedding_gene) - 1)]
                    gene_unlabel_i = dic_snps_gene[snps_unlabel_i]
                    unlabel_gene.append(dic_gene_emb[gene_unlabel_i])
                    unlabel_snps.append(dic_snps_emb[snps_unlabel_i])
                    unlabel_disease.append(dic_cui_emb[cuis_all[random.randint(0, len(cuis_all) - 1)]])

                    # train_pair.append(str(dic_snps_index[snps]) + "_"+cui)
                    #################wildtype########### hard negative#################
                    #################wildtype########### hard negative#################
                    gene = dic_snps_gene[snps]
                    traindata_cuis.append(dic_cui_emb[cui])
                    traindata_snps.append(dic_gene_emb[gene])

                    snps_p_temp = random.choice(dic_snp_positive_snps[snps])
                    traindata_snps_positive.append(dic_snps_emb[snps_p_temp])
                    traindata_snps_positive_gene.append(dic_gene_emb[dic_snps_gene[snps_p_temp]])
                    snps_N_temp = random.choice(dic_snp_negative_snps[snps])
                    traindata_snps_negative.append(dic_snps_emb[snps_N_temp])
                    traindata_snps_negative_gene.append(dic_gene_emb[dic_snps_gene[snps_N_temp]])

                    index_random = random.randint(0, len(traindata_cuis_P) - 1)
                    traindata_cuis_P.append(traindata_cuis_P[index_random])
                    traindata_snps_P.append(traindata_snps_P[index_random])
                    traindata_gene_P.append(traindata_gene_P[index_random])

                    train_gene.append(dic_gene_emb[gene])
                    train_gene_weight.append(dic_gene_weight[gene])

                    if snps in dic_snps_PPI_interface_positives:
                        snps_temp = random.choice(dic_snps_PPI_interface_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_interface_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_interface)
                    elif snps in dic_snps_domain_positives:
                        snps_temp = random.choice(dic_snps_domain_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_domain_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_domain)
                    elif snps in dic_snps_PPI_HINT_HQ_positives:
                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hint_hq)
                    elif snps in dic_snps_PPI_HINT_positives:
                        snps_temp = random.choice(dic_snps_PPI_HINT_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HINT_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hint)
                    elif snps in dic_snps_PPI_HIU_positives:
                        snps_temp = random.choice(dic_snps_PPI_HIU_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HIU_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hiu)
                    else:
                        train_gene_PPI_p1.append(dic_snps_emb[snps])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps]])
                        train_gene_PPI_p2.append(dic_gene_emb[gene])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[gene])
                        train_gene_PPI_p1_weight.append(1.0)

                    traindata_Y.append(0.0)
                    traindata_names.append(int(dic_snps_index[snps]))
                    # train_pair.append(str(dic_snps_index[snps]) + "_" + cui)
                    snps_unlabel_i = snps_with_embedding_gene[random.randint(0, len(snps_with_embedding_gene) - 1)]
                    gene_unlabel_i = dic_snps_gene[snps_unlabel_i]
                    unlabel_gene.append(dic_gene_emb[gene_unlabel_i])
                    unlabel_snps.append(dic_snps_emb[snps_unlabel_i])
                    dic_snps_unlabeled[snps_unlabel_i] = 1
                    dic_snps_unlabeled_gene[gene_unlabel_i] = 1
                    unlabel_disease.append(dic_cui_emb[cuis_all[random.randint(0, len(cuis_all) - 1)]])

                    gene = dic_snps_gene[snps]
                    dic_gene_train[gene] = 1
                    if flag_hard_mining > 0 and gene in dic_gene_emb:

                        traindata_cuis.append(dic_cui_emb["benign"])  # dic_gene_train[gene]=1
                        traindata_snps.append(dic_gene_emb[gene][0:768])

                        snps_p_temp = random.choice(dic_snp_positive_snps[snps])
                        traindata_snps_positive.append(dic_snps_emb[snps_p_temp])
                        traindata_snps_positive_gene.append(dic_gene_emb[dic_snps_gene[snps_p_temp]])
                        snps_N_temp = random.choice(dic_snp_negative_snps[snps])
                        traindata_snps_negative.append(dic_snps_emb[snps_N_temp])
                        traindata_snps_negative_gene.append(dic_gene_emb[dic_snps_gene[snps_N_temp]])

                        index_random = random.randint(0, len(traindata_cuis_P) - 1)
                        traindata_cuis_P.append(traindata_cuis_P[index_random])
                        traindata_snps_P.append(traindata_snps_P[index_random])
                        traindata_gene_P.append(traindata_gene_P[index_random])
                        train_gene.append(dic_gene_emb[gene])
                        train_gene_weight.append(dic_gene_weight[gene])

                        if snps in dic_snps_PPI_interface_positives:
                            snps_temp = random.choice(dic_snps_PPI_interface_positives[snps])
                            train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            snps_temp = random.choice(dic_snps_PPI_interface_negatives[snps])
                            train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            train_gene_PPI_p1_weight.append(weight_ppi_interface)
                        elif snps in dic_snps_domain_positives:
                            snps_temp = random.choice(dic_snps_domain_positives[snps])
                            train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            snps_temp = random.choice(dic_snps_domain_negatives[snps])
                            train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            train_gene_PPI_p1_weight.append(weight_domain)
                        elif snps in dic_snps_PPI_HINT_HQ_positives:
                            snps_temp = random.choice(dic_snps_PPI_HINT_HQ_positives[snps])
                            train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            snps_temp = random.choice(dic_snps_PPI_HINT_HQ_negatives[snps])
                            train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            train_gene_PPI_p1_weight.append(weight_ppi_hint_hq)
                        elif snps in dic_snps_PPI_HINT_positives:
                            snps_temp = random.choice(dic_snps_PPI_HINT_positives[snps])
                            train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            snps_temp = random.choice(dic_snps_PPI_HINT_negatives[snps])
                            train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            train_gene_PPI_p1_weight.append(weight_ppi_hint)
                        elif snps in dic_snps_PPI_HIU_positives:
                            snps_temp = random.choice(dic_snps_PPI_HIU_positives[snps])
                            train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            snps_temp = random.choice(dic_snps_PPI_HIU_negatives[snps])
                            train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                            train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                            train_gene_PPI_p1_weight.append(weight_ppi_hiu)
                        else:
                            train_gene_PPI_p1.append(dic_snps_emb[snps])
                            train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps]])
                            train_gene_PPI_p2.append(dic_gene_emb[gene])
                            train_gene_PPI_p2_gene.append(dic_gene_emb[gene])
                            train_gene_PPI_p1_weight.append(1.0)

                        traindata_Y.append(1.0)
                        traindata_names.append(int(dic_snps_index[snps]))

                        # train_pair.append(str(dic_snps_index[snps]) + "_" + cui)

                        snps_unlabel_i = snps_with_embedding_gene[random.randint(0, len(snps_with_embedding_gene) - 1)]
                        gene_unlabel_i = dic_snps_gene[snps_unlabel_i]
                        unlabel_gene.append(dic_gene_emb[gene_unlabel_i])
                        unlabel_snps.append(dic_snps_emb[snps_unlabel_i])
                        unlabel_disease.append(dic_cui_emb[cuis_all[random.randint(0, len(cuis_all) - 1)]])

                        dic_snps_unlabeled[snps_unlabel_i] = 1
                        dic_snps_unlabeled_gene[gene_unlabel_i] = 1

                        if gene in dic_gene_benign:  ############ ###if in benign: benigns snps or other pathogeneic SNPs ###############
                            dic_gene_train_in_benigh[gene] = 1
                            number_hard_negative_gene += 1
                            snps_benign = list(
                                set(list(dic_gene_benign[gene])) - set(snps_test))  # list(dic_gene_benign[gene])

                            snps_others = set(list(dic_gene_snps[gene])) - set(snps_test)
                            snps_benign_same_gene = list(snps_others)  # .intersection(set(snps_train)))
                            snps_benign_same_gene.extend(snps_benign)
                            snps_benign_same_gene = snps_benign
                            # snps_benign_same_gene.extend(snps_benign)
                            # snps_benign_same_gene.extend(snps_benign)
                            # snps_benign_same_gene.extend(snps_benign)
                            dic_gene_hit_as_benign[gene] = 0
                            num_temp = 0
                            valid_total = 1  # min(negative_num_max,int(0.1*len(snps_benign_same_gene)))
                            for samplei in range(30):
                                snps_b = snps_benign_same_gene[random.randint(0, len(snps_benign_same_gene) - 1)]
                                if num_temp < valid_total and not snps_b in dic_cui_snps[
                                    cui] and snps_b in dic_snps_emb and dic_snps_gene[snps_b] in dic_gene_emb:
                                    gene = dic_snps_gene[snps_b]
                                    num_temp += 1
                                    number_hard_negative += 1
                                    traindata_cuis.append(dic_cui_emb["benign"])  # dic_cui_emb["benign"]
                                    traindata_snps.append(dic_snps_emb[snps_b])

                                    snps_p_temp = random.choice(dic_snp_positive_snps[snps])
                                    traindata_snps_positive.append(dic_snps_emb[snps_p_temp])
                                    traindata_snps_positive_gene.append(dic_gene_emb[dic_snps_gene[snps_p_temp]])
                                    snps_N_temp = random.choice(dic_snp_negative_snps[snps])
                                    traindata_snps_negative.append(dic_snps_emb[snps_N_temp])
                                    traindata_snps_negative_gene.append(dic_gene_emb[dic_snps_gene[snps_N_temp]])

                                    index_random = random.randint(0, len(traindata_cuis_P) - 1)
                                    traindata_cuis_P.append(traindata_cuis_P[index_random])
                                    traindata_snps_P.append(traindata_snps_P[index_random])
                                    traindata_gene_P.append(traindata_gene_P[index_random])

                                    train_gene.append(dic_gene_emb[gene])
                                    if gene in dic_gene_weight:
                                        train_gene_weight.append(dic_gene_weight[gene])
                                    else:
                                        train_gene_weight.append(gene_weghts_median)

                                    if snps in dic_snps_PPI_interface_positives:
                                        snps_temp = random.choice(dic_snps_PPI_interface_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_PPI_interface_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_ppi_interface)
                                    elif snps in dic_snps_domain_positives:
                                        snps_temp = random.choice(dic_snps_domain_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_domain_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_domain)
                                    elif snps in dic_snps_PPI_HINT_HQ_positives:
                                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_ppi_hint_hq)
                                    elif snps in dic_snps_PPI_HINT_positives:
                                        snps_temp = random.choice(dic_snps_PPI_HINT_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_PPI_HINT_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_ppi_hint)
                                    elif snps in dic_snps_PPI_HIU_positives:
                                        snps_temp = random.choice(dic_snps_PPI_HIU_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_PPI_HIU_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_ppi_hiu)
                                    else:
                                        train_gene_PPI_p1.append(dic_snps_emb[snps])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps]])
                                        train_gene_PPI_p2.append(dic_gene_emb[gene])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[gene])
                                        train_gene_PPI_p1_weight.append(1.0)

                                    traindata_Y.append(1.0)
                                    traindata_names.append(int(dic_snps_index[snps]))

                                    snps_unlabel_i = snps_with_embedding_gene[
                                        random.randint(0, len(snps_with_embedding_gene) - 1)]
                                    gene_unlabel_i = dic_snps_gene[snps_unlabel_i]
                                    unlabel_gene.append(dic_gene_emb[gene_unlabel_i])
                                    unlabel_snps.append(dic_snps_emb[snps_unlabel_i])
                                    unlabel_disease.append(dic_cui_emb[cuis_all[random.randint(0, len(cuis_all) - 1)]])

                            num_temp = 0
                            for samplei in range(20):
                                snps_b = snps_benign_same_gene[random.randint(0, len(snps_benign_same_gene) - 1)]
                                if num_temp < 2 and not snps_b in dic_cui_snps[cui] and snps_b in dic_snps_emb and \
                                        dic_snps_gene[snps_b] in dic_gene_emb:
                                    num_temp += 1
                                    gene = dic_snps_gene[snps_b]
                                    traindata_cuis.append(dic_cui_emb[cui])
                                    traindata_snps.append(dic_snps_emb[snps_b])
                                    snps_p_temp = random.choice(dic_snp_positive_snps[snps])
                                    traindata_snps_positive.append(dic_snps_emb[snps_p_temp])
                                    traindata_snps_positive_gene.append(dic_gene_emb[dic_snps_gene[snps_p_temp]])
                                    snps_N_temp = random.choice(dic_snp_negative_snps[snps])
                                    traindata_snps_negative.append(dic_snps_emb[snps_N_temp])
                                    traindata_snps_negative_gene.append(dic_gene_emb[dic_snps_gene[snps_N_temp]])

                                    index_random = random.randint(0, len(traindata_cuis_P) - 1)
                                    traindata_cuis_P.append(traindata_cuis_P[index_random])
                                    traindata_snps_P.append(traindata_snps_P[index_random])
                                    traindata_gene_P.append(traindata_gene_P[index_random])

                                    index_random = random.randint(0, len(traindata_cuis_P) - 1)
                                    traindata_cuis_P.append(traindata_cuis_P[index_random])
                                    traindata_snps_P.append(traindata_snps_P[index_random])
                                    traindata_gene_P.append(traindata_gene_P[index_random])

                                    train_gene.append(dic_gene_emb[gene])
                                    if gene in dic_gene_weight:
                                        train_gene_weight.append(dic_gene_weight[gene])
                                    else:
                                        train_gene_weight.append(gene_weghts_median)

                                    if snps in dic_snps_PPI_interface_positives:
                                        snps_temp = random.choice(dic_snps_PPI_interface_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_PPI_interface_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_ppi_interface)
                                    elif snps in dic_snps_domain_positives:
                                        snps_temp = random.choice(dic_snps_domain_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_domain_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_domain)
                                    elif snps in dic_snps_PPI_HINT_HQ_positives:
                                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_ppi_hint_hq)
                                    elif snps in dic_snps_PPI_HINT_positives:
                                        snps_temp = random.choice(dic_snps_PPI_HINT_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_PPI_HINT_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_ppi_hint)
                                    elif snps in dic_snps_PPI_HIU_positives:
                                        snps_temp = random.choice(dic_snps_PPI_HIU_positives[snps])
                                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        snps_temp = random.choice(dic_snps_PPI_HIU_negatives[snps])
                                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                                        train_gene_PPI_p1_weight.append(weight_ppi_hiu)
                                    else:
                                        train_gene_PPI_p1.append(dic_snps_emb[snps])
                                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps]])
                                        train_gene_PPI_p2.append(dic_gene_emb[gene])
                                        train_gene_PPI_p2_gene.append(dic_gene_emb[gene])
                                        train_gene_PPI_p1_weight.append(1.0)

                                    traindata_Y.append(0.0)
                                    traindata_names.append(int(dic_snps_index[snps]))

                                    snps_unlabel_i = snps_with_embedding_gene[
                                        random.randint(0, len(snps_with_embedding_gene) - 1)]
                                    gene_unlabel_i = dic_snps_gene[snps_unlabel_i]
                                    unlabel_gene.append(dic_gene_emb[gene_unlabel_i])
                                    unlabel_snps.append(dic_snps_emb[snps_unlabel_i])
                                    unlabel_disease.append(dic_cui_emb[cuis_all[random.randint(0, len(cuis_all) - 1)]])

        if snps in snps_test:  ###test SNPs
            if len(cui_labeled) > 0:
                if flag_cross_cui > 0:
                    cui_labeled = set(dic_snps_cui[snps]) - CUI_train
                    cui_unlabeled = list(set(CUIs_all_covered) - set(cui_labeled) - set(CUI_train))
                else:
                    cui_labeled = set(dic_snps_cui[snps])
                    cui_unlabeled = list(set(CUIs_all_covered) - set(cui_labeled))

                for cui in cui_labeled:
                    eval_SNPs.append(int(dic_snps_index[snps]))
                    eval_index.append(CUIs_all_covered.index(cui))

                    gene = dic_snps_gene[snps]
                    eval_gene_test.append(dic_gene_emb[gene])

                    testdata_cuis.append(dic_cui_emb[cui])
                    testdata_snps.append(dic_snps_emb[snps])
                    gene = dic_snps_gene[snps]
                    testdata_gene.append(dic_gene_emb[gene])

                    testdata_Y.append(1.0)
                    testdata_names.append(snps)
                    test_pair.append(str(dic_snps_index[snps]) + "_" + cui)

                num_negative = 0
                for cui_i in range(negative_disease - 1):
                    cui = cui_unlabeled[random.randint(0, len(cui_unlabeled) - 1)]
                    num_negative = num_negative + 1
                    testdata_cuis.append(dic_cui_emb[cui])
                    testdata_snps.append(dic_snps_emb[snps])
                    gene = dic_snps_gene[snps]
                    testdata_gene.append(dic_gene_emb[gene])
                    testdata_Y.append(0.0)
                    testdata_names.append(snps)
                    test_pair.append(str(dic_snps_index[snps]) + "_" + cui)

                testdata_cuis.append(dic_cui_emb["benign"])
                testdata_snps.append(dic_snps_emb[snps])
                gene = dic_snps_gene[snps]
                testdata_gene.append(dic_gene_emb[gene])
                testdata_Y.append(0.0)
                testdata_names.append(snps)
                test_pair.append(str(dic_snps_index[snps]) + "_" + "benign")

    snps_total_labeled = set(list(dic_snps_cui.keys()))
    # snps_patho_all=set(snps_patho_all)-snps_total_labeled
    snps_patho_all_others = snps_patho_all - snps_total_labeled
    print("-------------------------snps_patho_all_others: ", len(snps_patho_all_others),
          "---------without traints but pathogenic for training---------")
    for snp in snps_patho_all_others:
        if True:
            gene = dic_snps_gene[snp]
            traindata_cuis.append(dic_cui_emb["benign"])  # dic_cui_emb["benign"]
            traindata_snps.append(dic_snps_emb[snp])

            snps_p_temp = random.choice(dic_snp_positive_snps[snp])
            traindata_snps_positive.append(dic_snps_emb[snps_p_temp])
            traindata_snps_positive_gene.append(dic_gene_emb[dic_snps_gene[snps_p_temp]])
            snps_N_temp = random.choice(dic_snp_negative_snps[snp])
            traindata_snps_negative.append(dic_snps_emb[snps_N_temp])
            traindata_snps_negative_gene.append(dic_gene_emb[dic_snps_gene[snps_N_temp]])

            index_random = random.randint(0, len(traindata_cuis_P) - 1)
            traindata_cuis_P.append(traindata_cuis_P[index_random])
            traindata_snps_P.append(traindata_snps_P[index_random])
            traindata_gene_P.append(traindata_gene_P[index_random])

            index_random = random.randint(0, len(traindata_cuis_P) - 1)
            traindata_cuis_P.append(traindata_cuis_P[index_random])
            traindata_snps_P.append(traindata_snps_P[index_random])
            traindata_gene_P.append(traindata_gene_P[index_random])

            train_gene.append(dic_gene_emb[gene])
            if gene in dic_gene_weight:
                train_gene_weight.append(dic_gene_weight[gene])
            else:
                train_gene_weight.append(gene_weghts_median)

            snps = snp
            if snps in dic_snps_PPI_interface_positives:
                snps_temp = random.choice(dic_snps_PPI_interface_positives[snps])
                train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                snps_temp = random.choice(dic_snps_PPI_interface_negatives[snps])
                train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                train_gene_PPI_p1_weight.append(weight_ppi_interface)
            elif snps in dic_snps_domain_positives:
                snps_temp = random.choice(dic_snps_domain_positives[snps])
                train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                snps_temp = random.choice(dic_snps_domain_negatives[snps])
                train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                train_gene_PPI_p1_weight.append(weight_domain)
            elif snps in dic_snps_PPI_HINT_HQ_positives:
                snps_temp = random.choice(dic_snps_PPI_HINT_HQ_positives[snps])
                train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                snps_temp = random.choice(dic_snps_PPI_HINT_HQ_negatives[snps])
                train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                train_gene_PPI_p1_weight.append(weight_ppi_hint_hq)
            elif snps in dic_snps_PPI_HINT_positives:
                snps_temp = random.choice(dic_snps_PPI_HINT_positives[snps])
                train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                snps_temp = random.choice(dic_snps_PPI_HINT_negatives[snps])
                train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                train_gene_PPI_p1_weight.append(weight_ppi_hint)
            elif snps in dic_snps_PPI_HIU_positives:
                snps_temp = random.choice(dic_snps_PPI_HIU_positives[snps])
                train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                snps_temp = random.choice(dic_snps_PPI_HIU_negatives[snps])
                train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                train_gene_PPI_p1_weight.append(weight_ppi_hiu)
            else:
                train_gene_PPI_p1.append(dic_snps_emb[snps])
                train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps]])
                train_gene_PPI_p2.append(dic_gene_emb[gene])
                train_gene_PPI_p2_gene.append(dic_gene_emb[gene])
                train_gene_PPI_p1_weight.append(1.0)

            traindata_Y.append(0.0)
            traindata_names.append(snp)
            snps_unlabel_i = snps_with_embedding_gene[random.randint(0, len(snps_with_embedding_gene) - 1)]
            gene_unlabel_i = dic_snps_gene[snps_unlabel_i]
            unlabel_gene.append(dic_gene_emb[gene_unlabel_i])
            unlabel_snps.append(dic_snps_emb[snps_unlabel_i])
            unlabel_disease.append(dic_cui_emb[cuis_all[random.randint(0, len(cuis_all) - 1)]])

    ################negative sampling based on diseases#############
    for cui in dic_cui_snps:
        snps_labeled = dic_cui_snps[cui]
        snps_unlabeled = set(snps_train) - set(snps_labeled)
        snps_benign_all = set(list(dic_snps_benign.keys()))
        snps_unlabeled = snps_benign_all  # snps_unlabeled.union(snps_benign_all)

        if cui in CUI_train:
            num_negative = 0
            for snps in random.sample(snps_benign_all, negative_snps):
                if dic_snps_gene[snps] in dic_gene_emb:
                    num_negative = num_negative + 1
                    traindata_cuis.append(dic_cui_emb[cui])
                    traindata_snps.append(dic_snps_emb[snps])

                    snps_p_temp = random.choice(dic_snp_positive_snps[snps])
                    traindata_snps_positive.append(dic_snps_emb[snps_p_temp])
                    traindata_snps_positive_gene.append(dic_gene_emb[dic_snps_gene[snps_p_temp]])
                    snps_N_temp = random.choice(dic_snp_negative_snps[snps])
                    traindata_snps_negative.append(dic_snps_emb[snps_N_temp])
                    traindata_snps_negative_gene.append(dic_gene_emb[dic_snps_gene[snps_N_temp]])

                    index_random = random.randint(0, len(traindata_cuis_P) - 1)
                    traindata_cuis_P.append(traindata_cuis_P[index_random])
                    traindata_snps_P.append(traindata_snps_P[index_random])
                    traindata_gene_P.append(traindata_gene_P[index_random])

                    index_random = random.randint(0, len(traindata_cuis_P) - 1)
                    traindata_cuis_P.append(traindata_cuis_P[index_random])
                    traindata_snps_P.append(traindata_snps_P[index_random])
                    traindata_gene_P.append(traindata_gene_P[index_random])

                    gene = dic_snps_gene[snps]
                    train_gene.append(dic_gene_emb[gene])
                    if gene in dic_gene_weight:
                        train_gene_weight.append(dic_gene_weight[gene])
                    else:
                        train_gene_weight.append(gene_weghts_median)

                    if snps in dic_snps_PPI_interface_positives:
                        snps_temp = random.choice(dic_snps_PPI_interface_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_interface_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_interface)
                    elif snps in dic_snps_domain_positives:
                        snps_temp = random.choice(dic_snps_domain_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_domain_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_domain)
                    elif snps in dic_snps_PPI_HINT_HQ_positives:
                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HINT_HQ_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hint_hq)
                    elif snps in dic_snps_PPI_HINT_positives:
                        snps_temp = random.choice(dic_snps_PPI_HINT_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HINT_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hint)
                    elif snps in dic_snps_PPI_HIU_positives:
                        snps_temp = random.choice(dic_snps_PPI_HIU_positives[snps])
                        train_gene_PPI_p1.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        snps_temp = random.choice(dic_snps_PPI_HIU_negatives[snps])
                        train_gene_PPI_p2.append(dic_snps_emb[snps_temp])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[dic_snps_gene[snps_temp]])
                        train_gene_PPI_p1_weight.append(weight_ppi_hiu)
                    else:
                        train_gene_PPI_p1.append(dic_snps_emb[snps])
                        train_gene_PPI_p1_gene.append(dic_gene_emb[dic_snps_gene[snps]])
                        train_gene_PPI_p2.append(dic_gene_emb[gene])
                        train_gene_PPI_p2_gene.append(dic_gene_emb[gene])
                        train_gene_PPI_p1_weight.append(1.0)

                    traindata_Y.append(0.0)
                    traindata_names.append(traindata_names[0])
                    # train_pair.append(str(dic_snps_index[snps]) + "_" + cui)

                    snps_unlabel_i = snps_with_embedding_gene[random.randint(0, len(snps_with_embedding_gene) - 1)]
                    gene_unlabel_i = dic_snps_gene[snps_unlabel_i]
                    unlabel_gene.append(dic_gene_emb[gene_unlabel_i])
                    unlabel_snps.append(dic_snps_emb[snps_unlabel_i])
                    unlabel_disease.append(dic_cui_emb[cuis_all[random.randint(0, len(cuis_all) - 1)]])

                    dic_snps_unlabeled[snps_unlabel_i] = 1
                    dic_snps_unlabeled_gene[gene_unlabel_i] = 1

        if cui in CUI_test:
            snps_unlabeled = set(snps_test) - set(snps_labeled)
            num_negative = 0
            for snps in random.sample(snps_unlabeled, min(int(negative_snps / 2), int(0.2 * len(snps_unlabeled)))):
                if dic_snps_gene[snps] in dic_gene_emb:
                    num_negative = num_negative + 1
                    testdata_cuis.append(dic_cui_emb[cui])
                    testdata_snps.append(dic_snps_emb[snps])
                    gene = dic_snps_gene[snps]
                    testdata_gene.append(dic_gene_emb[gene])
                    testdata_Y.append(0.0)
                    testdata_names.append(snps)
                    test_pair.append(str(snps) + "_" + cui)

            for snps in random.sample(snps_benign_all, int(negative_snps / 2)):
                if dic_snps_gene[snps] in dic_gene_label and dic_snps_gene[snps] in dic_gene_emb:
                    num_negative = num_negative + 1
                    gene = dic_snps_gene[snps]
                    testdata_cuis.append(dic_cui_emb[cui])
                    testdata_snps.append(dic_snps_emb[snps])
                    testdata_gene.append(dic_gene_emb[gene])
                    testdata_Y.append(0.0)
                    testdata_names.append(snps)
                    test_pair.append(str(snps) + "_" + cui)


    return (traindata_snps, traindata_cuis, traindata_snps_P, traindata_cuis_P, traindata_gene_P, traindata_Y, traindata_names,
    testdata_cuis, testdata_snps, testdata_Y, testdata_names, test_pair, eval_SNPs, eval_index,
    train_gene_PPI_p2, train_gene_PPI_p2_gene, unlabel_disease, eval_SNPs_train,
    eval_cui_train, train_gene, train_gene_PPI_p1, train_gene_PPI_p1_gene, testdata_gene,
    eval_gene_train, eval_gene_test, train_gene_weight,
    traindata_snps_positive, traindata_snps_positive_gene, traindata_snps_negative, traindata_snps_negative_gene,
    train_gene_PPI_p1_weight)


if __name__ == '__main__':
    print("loaddding data begin...")
    loaddata()
    print("loaddding data end...")












