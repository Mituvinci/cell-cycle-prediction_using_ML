sed '1d' GSE146773_Counts.csv|cut -f 1 -d ',' > barcodes.tsv

head -1 GSE146773_Counts.csv |tr ',' '\n'|sed '1d' > features.tsv



sed '1d' GSE146773_Counts.csv |tr ',' '\t'|awk '{for (i=2;i<=NF;i++) print NR,i-1,$i}' |grep -v '0.00'|awk '{print $2,$1,int($3)}' OFS=' ' |sort -k 2,2n -k1,1n > GSE146773_Counts.matrix.csv



# after get header

# header need matrix.csv file information
# feature number
# barcode number
# total number


sh write.header.sh


cat header.txt GSE146773_Counts.matrix.csv > matrix.mtx




