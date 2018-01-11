
Hycco
-----

Hycco is a Python program that uses an HMM based method to estimate HYbrid
Chromosomal CrossOver points for *haploid* samples using distinguishing SNPs
from two parental genotypes. The method works by analysing SNP variations for a
hybrid sample (e.g. from chromosomal crossover after meiosis) that has been
compared TWO separate parental genotypes.

Each chromosome is first segmented into binned regions of a specified length
(typically several Mb, depending on SNP density) and the presence of SNPs that
can distinguish one parental genotype from the other are used to train a
Gaussian HMM using the Baum-Welch method that is then interrogated for the
probabilities of its underlying (hidden) parental genotype states (labelled as A
or B) using the Forward-Backward method. This attempts to assign one of the two
parental genotype states (A or B) to contiguous segments of the chromosomes.
Where the genotype state swaps between A and B a chromosomal crossover is
inferred. The crossover point is then more precisely estimated, at a resolution
better than the initial HMM segments. Here the closest pair of parental specific
SNPs to the HMM segment edge that swap from A to B or from B to A (in the same
way as the HMM state change) are sought. The crossover point is then estimated
as halfway between these two A/B distinguishing SNPs.

This method only uses SNPs and will ignore larger variations listed in the VCF
files. Also, detecting hybrid crossover in this way is, naturally, dependent on
having enough SNPs that are specific to either the A or B genotype; if there is
a region of chromosomal crossover with no parent-specific SNP it will not be
detected. If a chromosomal region has no parental-specific SNPs (e.g. unmappable
regions) or roughly equal numbers from both parents the optimum state will be
poorly defined and the predicted A/B genotype label from the HMM can arbitrary
flip between A and B. Such regions will generally give a low probability of a
parental genotype (as illustrated via the grey line in graph output) and so are
deemed unreliable. While low state probabilities are expected at A/B
transitional segments, if they occur over long regions a smaller bin/segment
size (-b flag) may be appropriate to catch small crossover regions, if there is
enough SNP data. However, choosing smaller bin sizes will tend to increase the
number of SNP-free segments and thus increase noise in the A/B genotype
assignment.

Notes
-----

Hycco does not currently work with multi-sample VCF files.

Hycco only works with VCF datasets representing a single haploid sample. It does
not work with haploid mixtures and diploids etc.

Hycco works with both Python 2 (version >= 2.6) and Python 3.


Dependencies
------------

Using Hycco is dependent upon installation of the following:

* Python >= 2.6
* NumPy
* SciPy
* scikit-learn
* hmmlearn

Python modules may be readily installed via the pip mechanism, e.g:

  `sudo pip install sklearn sudo pip install hmmlearn`

or for a single user:

  `pip install -U --user sklearn pip install -U --user hmmlearn`

  
Installation
------------

Caveat to the above mentioned dependencies being installed, Hycco itself
requires no special installation any may be run directly from its directory once
it has been downloaded (e.g. using the "Clone or download" option in GitHub). 


Running Hycco
-------------

Hycco is run by using the "hycco" command that represents an executable script,
which in turn runs the hycco.py Python script. The "hycco" script must have
executable file permission and either be available on the system's executable
path (e.g. $PATH) or be run using a specific directory path, such as ./hycco or
/usr/bin/hycco.

Typical use:

  `./hycco hybrid_to_genome_A.vcf hybrid_to_genome_B.vcf`


Input files
-----------

The input files should be of [VCF
format](http://www.internationalgenome.org/wiki/Analysis/vcf4.0/) and should be
specified in pairs, i.e. ordered so that paired files are listed next to one
another. Each pair of VCF files should correspond to one haploid hybrid sample
where the individual VCF files of the pair correspond to variant calls for two
different parental genotypes.

For example, for two sequenced hybrid samples named "sample_1" and sample
"sample_2" that have been mapped to two genotypes named "genome_A" and
"genome_B" there will be four input VCF files, where each sample is paired with
each genotype. These may be used with Hycco using a command like:

  `./hycco sample_1_genome_A.vcf sample_1_genome_B.vcf sample_2_genome_A.vcf sample_2_genome_B.vcf`


The input VCF files would typically be generated by first mapping FASTQ sequence
reads for each hybrid sample containing to the two reference genome builds using
an aligner such as [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2) or
[BWA](https://github.com/lh3/bwa), to make BAM/SAM files, and then creating VCF
files of variant calls using programs such as
[freebayes](https://github.com/ekg/freebayes) or
[GATK](https://software.broadinstitute.org/gatk/).


Output format
-------------

The output files are generated in a tab-separated text format which may be read
into most spreadsheet programs, using a CSV or similar import option.

The first line of the output file (beginning '#HMM') gives the input parameters
that Hycco used to generate the data. E.g:

#HMM params - bin_size:5000 min_qual:200 min_chromo_size:1000000 num_hmm_iter:400


The next line (beginning '#cols') describes the column headings:

#cols - chr  haplotype  region_size  region_start  region_end first_SNP  last_SNP  bin_start bin_end


The columns respectively describe:

* Chromosome/contig name
* Parental haploid genotype state: A or B, for a crossed-over region
* Estimated bp size for the chromosome region
* Estimated region bp start point (either a chromosome edge or between discriminating SNPs)
* Estimated region bp end point
* Bp position of the first parental-discriminating SNP that matches the region's genotype state 
* Bp position of the last parental-discriminating SNP that matches the region's genotype state  
* Bp position of the first binned HMM segment that defined this region 
* Bp position of the last binned HMM segment that defined this region

The data lines, one corresponding to each consecutive A or B genotype region,
then follow using the described column order.


Command line options for Hycco
------------------------------

sage: Hycco [-h] [-b BIN_SIZE] [-g] [-m MIN_CHROMO_SIZE] [-n NUM_HMM_ITER]
             [-o OUT_DIR] [-q] [-s RANDOM_SEED]
             [-t TEXT_LABEL [TEXT_LABEL ...]]
             VCF_FILE [VCF_FILE ...]

positional arguments:
  VCF_FILE              Input VCF format files containing variant calls for
                        each parental genotype/strain. Files should be listed
                        sequentially in pair order. Inputs may be gzipped
                        (this is assumed by a .gz file extension).

optional arguments:
  -h, --help            show this help message and exit
  -b BIN_SIZE           Binned analysis region size (in bp) for defining HMM
                        chromosome segments (). Default: 10000 bp.
  -g                    Specifies that graphical output will be displayed for
                        each VCF pair using matplotlib
  -m MIN_CHROMO_SIZE    Minimum length (in Mb) required for a
                        contig/chromosome (as described in VCF header) to be
                        analysed. Default: 1.00 Mb.
  -n NUM_HMM_ITER       Number of iterations to perform when estimating
                        Gaussian HMM probabilities using the Baum-Welch
                        method. Default: 400
  -o OUT_DIR            Optional output directory for writing results.
                        Defaults to the current working directory.
  -q                    Minimum quality (phred-scale) for accepting a SNP
                        call, as described in the VCF data lines. Default:
                        100.00
  -s RANDOM_SEED        Optional random seed value, i.e. to make repeat
                        calculations deterministic.
  -t TEXT_LABEL [TEXT_LABEL ...]
                        Optional text labels to describe each input pair,
                        which are used to name output files as
                        {NAME}_crossover_regions.tsv, in the same order as the
                        input VCF file pairs.

For further help email tstevens@mrc-lmb.cam.ac.uk or garethb@mrc-lmb.cam.ac.uk