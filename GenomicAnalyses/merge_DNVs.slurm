#!/bin/bash
#SBATCH -N1 --exclusive
#SBATCH -t 1-00:0
#SBATCH --mail-user=EMAIL_ADDRESS
#SBATCH --mail-type=END,FAIL

# Directory containing VCF files
input_dir="/path/to/variant/calls/"

# Output directory for compressed files
output_dir="compressed_vcfs_filtered/"

# Output directory for merged file
merged_output="DNV_calls_WES_v3_merged_filtered.vcf.gz"

# Create output directory for compressed files
mkdir -p "$output_dir"

# Compress each VCF file with bgzip
for vcf_file in "$input_dir"/*/*.glnexus.family.combined_intersection_filtered_gq_20_depth_10_position.vcf; do
    # Get the base name without the extension
    base_name="$(basename -s .vcf "$vcf_file")"
    
    # Compress the VCF file with bgzip
    bgzip -c "$vcf_file" > "$output_dir/$base_name.vcf.gz"
    # index the compressed VCF file with tabix
    tabix -p vcf "$output_dir/$base_name.vcf.gz"
done

# Merge compressed VCF files with bcftools
bcftools merge --merge none -O z --force-samples "$output_dir"/*.vcf.gz -o "$merged_output"
