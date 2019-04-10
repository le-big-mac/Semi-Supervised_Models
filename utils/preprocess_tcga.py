import pandas as pd


rnaseq_df = pd.read_table('../data/tcga/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', index_col=0)
sample_labels_df = pd.read_csv('../data/tcga/labels.csv')

# appears twice in the dataset, drop both
rnaseq_df.drop('SLC35E2', axis=0, inplace=True)

rnaseq_df = rnaseq_df.T

sample_labels_df = sample_labels_df[['SAMPLE_BARCODE', 'DISEASE']]
sample_labels_df = sample_labels_df.set_index('SAMPLE_BARCODE')

sample_barcodes = set(sample_labels_df.index)
sample_barcodes = sample_barcodes.intersection(set(rnaseq_df.index))

rnaseq_df = rnaseq_df.loc[sample_barcodes, :]
rnaseq_df = rnaseq_df[~rnaseq_df.index.duplicated()]

rnaseq_df = pd.merge(rnaseq_df, sample_labels_df, left_index=True, right_index=True)
rnaseq_df = rnaseq_df.sort_values('DISEASE')

rnaseq_df.to_csv('../data/tcga/rnaseq_data_with_labels.csv')