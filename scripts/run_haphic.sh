#!/usr/bin/env bash
set -euo pipefail

############################
# User settings
############################
RAW_FA="/jdh/SRA/hifiasm_runs/Vitis_vinifera/SRR29686483.asm.bp.p_ctg.fa"
HYBRID_FA="/jdh/SRA/hifiasm_runs/Vitis_vinifera/organelle_cleaner_runs/hybrid/cleaned_assembly.fa"

R1="/jdh/SRA/oryza_hic/SRR29686480_1.fastq.gz"
R2="/jdh/SRA/oryza_hic/SRR29686480_2.fastq.gz"

OUTROOT="/jdh/SRA/hifiasm_runs/Vitis_vinifera/haphic_compare"
THREADS=128
SAMTOOLS_THREADS=40
FILTER_THREADS=128
NCHR=19

HAPHIC="/jdh/tool/HapHiC/haphic"
FILTER_BAM="/jdh/tool/HapHiC/utils/filter_bam.py"
CONDA_SH="/home/jdh/miniconda3/etc/profile.d/conda.sh"

############################
# Load conda env
############################
source "${CONDA_SH}"
conda activate haphic

mkdir -p "${OUTROOT}"

run_haphic () {
    local label="$1"
    local fasta="$2"

    local workdir="${OUTROOT}/${label}"
    mkdir -p "${workdir}"
    cd "${workdir}"

    echo "=================================================="
    echo "[INFO] Running HapHiC for: ${label}"
    echo "[INFO] Assembly: ${fasta}"
    echo "[INFO] Workdir : ${workdir}"
    echo "=================================================="

    ############################
    # 1. BWA index
    ############################
    if [[ ! -f "${fasta}.bwt" ]]; then
        echo "[INFO] Building BWA index..."
        bwa index "${fasta}"
    else
        echo "[INFO] BWA index already exists. Skipping."
    fi

    ############################
    # 2. Mapping Omni-C reads
    ############################
    if [[ ! -f "${workdir}/HiC.bam" ]]; then
        echo "[INFO] Mapping Omni-C reads..."
        bwa mem -5SP -t "${THREADS}" "${fasta}" "${R1}" "${R2}" \
            | samblaster \
            | samtools view - -@ "${SAMTOOLS_THREADS}" -S -h -b -F 3340 -o "${workdir}/HiC.bam"
    else
        echo "[INFO] HiC.bam already exists. Skipping mapping."
    fi

    ############################
    # 3. BAM filtering
    ############################
    if [[ ! -f "${workdir}/HiC.filtered.bam" ]]; then
        echo "[INFO] Filtering BAM..."
        "${FILTER_BAM}" "${workdir}/HiC.bam" 1 --NM 3 --threads "${FILTER_THREADS}" \
            | samtools view - -b -@ "${SAMTOOLS_THREADS}" -o "${workdir}/HiC.filtered.bam"
    else
        echo "[INFO] HiC.filtered.bam already exists. Skipping filtering."
    fi

    ############################
    # 4. HapHiC pipeline
    ############################
    if [[ ! -d "${workdir}/04.build" ]]; then
        echo "[INFO] Running HapHiC pipeline..."
        "${HAPHIC}" pipeline "${fasta}" "${workdir}/HiC.filtered.bam" "${NCHR}" \
            --threads "${THREADS}" --processes "${THREADS}"
    else
        echo "[INFO] 04.build directory already exists. Skipping pipeline."
    fi

    ############################
    # 5. Juicebox script
    ############################
    if [[ -d "${workdir}/04.build" ]]; then
        cd "${workdir}/04.build"
        if [[ -f "juicebox.sh" ]]; then
            echo "[INFO] Running juicebox.sh..."
            bash juicebox.sh
        else
            echo "[WARN] juicebox.sh not found in ${workdir}/04.build"
        fi
    else
        echo "[WARN] 04.build directory not found, skipping juicebox."
    fi

    ############################
    # 6. HapHiC plot
    ############################
    if [[ -f "${workdir}/04.build/scaffolds.raw.agp" ]]; then
        echo "[INFO] Plotting contact map..."
        "${HAPHIC}" plot "${workdir}/04.build/scaffolds.raw.agp" "${workdir}/HiC.filtered.bam"
    else
        echo "[WARN] scaffolds.raw.agp not found, skipping plot."
    fi

    echo "[DONE] ${label}"
    echo
}

run_haphic "raw" "${RAW_FA}"
run_haphic "hybrid" "${HYBRID_FA}"

echo "All HapHiC runs completed."
