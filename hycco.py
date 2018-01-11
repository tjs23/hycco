import os, re, math, sys
from collections import defaultdict

VCF_CONTIG_PATT = re.compile('ID=(\w+),length=(\d+)')

PROG_NAME = 'Hycco'

DESCRIPTION = 'Hycco is an HMM based method to estimate hybrid chromosomal crossover points using distinguising SNPs from two parental genotypes'

FILE_TAG = 'crossover_regions'

DEFAULT_BIN_SIZE = 10000

DEFAULT_QUALITY = 100.0

DEFAULT_MIN_CHROMO_SIZE = 1.0

DEFAULT_NUM_ITER = 400

def info(msg, prefix='INFO'):
  print('%8s : %s' % (prefix, msg))


def warn(msg, prefix='WARNING'):
  print('%8s : %s' % (prefix, msg))


def fatal(msg, prefix='%s FAILURE' % PROG_NAME):
  print('%8s : %s' % (prefix, msg))
  sys.exit(0)


def check_file_problems(file_path):

  problem = None
  
  if not os.path.exists(file_path):
    problem = 'File "%s" does not exist'
    return problem % file_path
  
  if not os.path.isfile(file_path):
    problem = 'Location "%s" is not a regular file'
    return problem % file_path
  
  if os.stat(file_path).st_size == 0:
    problem = 'File "%s" is of zero size '
    return problem % file_path
    
  if not os.access(file_path, os.R_OK):
    problem = 'File "%s" is not readable'
    return problem % file_path
  
  return problem


def test_imports():
  
  critical = False

  try:
    from numpy import array
  except ImportError:
    critical = True
    warn('Critical module "numpy" is not installed or accessible')

  try:
    from sklearn import cluster
  except ImportError:
    critical = True
    warn('Critical module "sklearn" is not installed or accessible')

  try:
    from hmmlearn import hmm
  except ImportError:
    critical = True
    warn('Critical module "hmmlearn" is not installed or accessible')

  try:
    from matplotlib import pyplot
  
  except ImportError:
    warn('Module "matplotlib" is not installed or accessible. Graphing option is not available.')  
  
  if critical:
    fatal('Exiting because critial Python modules are not available')
    

test_imports()
    
import numpy as np

def read_vcf(file_path, min_qual=100):
  """
  Read VCF file to get SNPs with a given minimum quality.
  Reurns chromsome sizes and SPN positions with qualities, keyed by chromosome
  """
  
  if file_path.lower().endswith('.gz'):
    import gzip
    file_obj = gzip.open(file_path, 'rt')
  else:
    file_obj = open(file_path)
  
  chromo_dict = {}
  var_dict = defaultdict(list)
  
  for line in file_obj:
    if line[:9] == '##contig=':
      match = VCF_CONTIG_PATT.search(line)
      
      if match:
        chromo = match.group(1)
        size = int(match.group(2))
        chromo_dict[chromo] = size

    elif line[0] == '#':
      continue
   
    else:
      chromo, pos, _id, ref, alt, qual, filt, info, fmt, xgb3 = line.split()
      qual = float(qual)
      
      if len(ref) == 1 and len(alt) == 1:
        if qual >= min_qual:
          var_dict[chromo].append((int(pos), qual))
  
  file_obj.close()
  
  # VCF may not be sorted
  sort_var_dict = {}
  for chromo in var_dict:
    sort_var_dict[chromo] = sorted(var_dict[chromo])
         
  return chromo_dict, sort_var_dict


def get_contig_data(chromo_dict, var_dict, bin_size=1000):
  """
  Given a dictionary of chromosome sizes, bins the log_10 scores of variant
  positions into a contiguous array for each chromosome. Returns a list
  of lists, one for each chromosome in sorted order.
  """
  
  contigs = []
  bs = float(bin_size)
  
  for chromo in sorted(chromo_dict):
    size = chromo_dict[chromo]
    n = int(math.ceil(size/bs)) + 1
    contig = np.zeros(n, float)
    
    for pos, qual in var_dict.get(chromo, []):
      i = int(pos/bin_size)
      w = (pos - (i*bin_size))/bs
      q =  np.log10(1.0 + qual)
      
      contig[i] += (1.0-w) * q
      contig[i+1] += w * q
      
    contigs.append(contig)
  
  return contigs
      

def get_training_data(contig_data1, contig_data2):
  """
  Joins the TMM training data for contiguous SNP scores derived from
  different genome builds.
  """
  
  data = []
  sizes = []
  
  for i, contig1 in enumerate(contig_data1):
    contig2 = contig_data2[i]
    
    row = list(zip(contig1, contig2))
    sizes.append(len(row))
    data.append(np.array(row))
  
  data = np.concatenate(data)
  
  return data, sizes
    
  
    
def train_hybrid_crossover_hmm(vcf_paths_pairs, text_labels=('A','B'), out_dir='', bin_size=10000, min_qual=100,
                              min_chromo_size=1e6, num_hmm_iter=400, plot_graphs=False, covariance_type='diag'):
  """
  Main function to train the HMM and plot the results
  """
  
  from hmmlearn import hmm
  
  # This is simply to remove worrysome messages which ough to be addressed in newer hmmlearn versions
  from sklearn import warnings
  def nullfunc(*args, **kw):
    pass
    
  warnings.warn = nullfunc
  
  chromos = set()
  var_pairs = []
  chromo_dicts = []
  n_states=2
  
  # Read the VCF data and chromosome sizes
  
  for vcf_path_a, vcf_path_b in vcf_paths_pairs:
    chromo_dict_a, var_dict_a = read_vcf(vcf_path_a, min_qual)
    chromo_dict_b, var_dict_b = read_vcf(vcf_path_b, min_qual)
  
    chromos.update(chromo_dict_a.keys())
    chromos.update(chromo_dict_b.keys())
    
    var_pairs.append((var_dict_a, var_dict_b))
    chromo_dicts += [chromo_dict_a, chromo_dict_b]
  
  # Collate chromosome sizes, talking the largest,
  # just in case there are any discrepencies and
  # ignoring any that are too small to bother with
  
  chromo_dict = {}
  for chromo in chromos:
    size = max([cd.get(chromo, 0) for cd in chromo_dicts])
    
    if size >= min_chromo_size:
      chromo_dict[chromo] = size
  
  chromos = sorted(chromo_dict)
  
  # Look through variant call pairs for each strain
  
  if plot_graphs:
    fig, axarr = plt.subplots(len(chromos), len(var_pairs))
    title_text = 'Hybrid genome HMM states. Bin size = {:,} Min qual = {}'
    fig.suptitle(title_text.format(bin_size, min_qual))
  
  n_cols = len(var_pairs)
  
  head_1 = '#HMM params - bin_size:%d min_qual:%d min_chromo_size:%d num_hmm_iter:%d\n'
  head_2 = '#cols - chr\thaplotype\tregion_size\tregion_start\tregion_end\tfirst_SNP\tlast_SNP\tbin_start\tbin_end\n'
  
  for col, (var_dict_a, var_dict_b) in enumerate(var_pairs):
    file_name = '%s_%s.tsv' % (text_labels[col], FILE_TAG)
    file_path = os.path.join(out_dir, file_name)
    file_obj = open(file_path, 'w')
    write = file_obj.write

    write(head_1 % (bin_size, min_qual, min_chromo_size, num_hmm_iter))
    write(head_2)
   
    contig_data_a = get_contig_data(chromo_dict, var_dict_a, bin_size)
    contig_data_b = get_contig_data(chromo_dict, var_dict_b, bin_size)
    in_data, sizes = get_training_data(contig_data_a, contig_data_b)
    
    # Setup an HMM object
    model = hmm.GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                            n_iter=num_hmm_iter)
    
    # Run Baum-Welch to lear the HMM probabilities
    
    model.fit(in_data, sizes)
    
    mv = in_data.max()
    i = 0
    
    for row, chromo in enumerate(chromos):
      m = sizes[row]
      chrom_data = in_data[i:i+m]
      i += m
      
      # Run Forward-Backward to get state probabilities at each point
      probs = model.predict_proba(chrom_data, [m]) 
    
      # The order of state labels is arbitrary, so use a dot product to 
      # deduce which state best matches the first genome
      dp1 = np.dot(probs[:,0], chrom_data[:,0])
      dp2 = np.dot(probs[:,0], chrom_data[:,1])
 
      if dp2 > dp1:
        probs_a = probs[:,1]
        probs_b = probs[:,0]
      else:
        probs_a = probs[:,0]
        probs_b = probs[:,1]
      
      # Create chromosome regios of contiguous state according to which of
      # the probabilities for the binned regions was higest
      
      prev_state = 0
      region_start = 0
      chromo_regions = []
      for j in range(m):
        pos = j * bin_size
      
        if probs_a[j] > probs_b[j]:
          state = 'A'
        elif probs_a[j] < probs_b[j]:
          state = 'B'
        else:
          state = ''
      
        if state != prev_state:
          if prev_state:
            chromo_regions.append((region_start, pos, prev_state))
          region_start = pos
        
        prev_state = state
      
      # Last region goes to the chromosome end
      if state:
        chromo_regions.append((region_start, min(pos, chromo_dict[chromo]), prev_state))
      
      
      # Refine region edges according to precise SNP positions
      # which could be before or after end of binned region
      
      # Remove SNPs common to both genotypes
      pos_counts = defaultdict(int)
      for pos, qual in var_dict_a[chromo]:
        pos_counts[pos] += 1
        
      for pos, qual in var_dict_b[chromo]:
        pos_counts[pos] += 1
      
      # Get sorted positions (and corresponding states) of distinguishing SNPs
      
      vars_state_pos  = [(pos, 'A') for pos, qual in var_dict_a[chromo] if pos_counts[pos] < 2]
      vars_state_pos += [(pos, 'B') for pos, qual in var_dict_b[chromo] if pos_counts[pos] < 2]
      vars_state_pos.sort()
      
      var_pos, var_states = zip(*vars_state_pos)
      var_pos = np.array(var_pos)
      n_var = len(var_pos)
      
      for start, end, state in chromo_regions:

        # Find transitions from A/B genotypes SNPs working away from region edge
        # Report terminal snips and mid-way between transitions, where possible
         
        # Refine beginning
         
        idx_left = idx_right = np.abs(var_pos-start).argmin() # Closest SNP to bin boundary
          
        while (idx_right < n_var-1) and (var_states[idx_right] != state): # Find next matching SNP
          idx_right += 1
          
        while (idx_left >= 0) and (var_states[idx_left] == state): # Find prev mismatching SNP
          idx_left -= 1
          
        vp1 = var_pos[idx_right]
        
        if vp1 > end:
          msg = 'No SNPs for HMM state "%s" found in chromosome region %s:%d-%d. '
          msg += 'Probably the HMM was not able to separate states in the data as expected'
          warn(msg)
          vp1 = start
        
        if idx_left < 0: # Off the chromosome start
          rp1 = 0
        else:
          rp1 = int((var_pos[idx_left] + vp1)/2)
        
        # Refine end
        
        idx_left = idx_right = np.abs(var_pos-end).argmin() # Closest SNP to bin boundary
          
        while (idx_left >= 0) and (var_states[idx_left] != state): # Find prev matching SNP
          idx_left -= 1
          
        while (idx_right < n_var) and (var_states[idx_right] == state): # Find next mismatching SNP
          idx_right += 1
          
        vp2 = var_pos[idx_left]
        
        if vp2 < start:
          vp2 = end
         
        if idx_right < n_var:
          rp2 = int((vp2 + var_pos[idx_right])/2)
        else: # Off the chromosome end
          rp2 = end
                
        # Chromosome, state code, region start, region end, region length,
        #   first matching var position, last matching var pos
        line_data = (chromo, state, rp2-rp1, rp1, rp2, vp1, vp2, start, end)
        line = '%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' % line_data
        write(line)
      
      if plot_graphs:
        # Plot first probabilities at a resonable scale
        probs = probs_a * 0.75 * mv
 
        # X valuys for plot in Megabases
        x_vals = np.array(range(len(chrom_data))) * bin_size / 1e6
        nx = x_vals[-1]
 
        # Plot the lines
        
        if n_cols > 1:
          ax = axarr[row, col]
        else:
          ax = axarr[row]
        
        ax.plot(x_vals, chrom_data[:,0], color='#FF4000', alpha=0.4, linewidth=1.5, label='Genome A SNPs')
        ax.plot(x_vals, chrom_data[:,1], color='#0080FF', alpha=0.4, linewidth=1.5, label='Genome B SNPs')
        ax.plot(x_vals, probs[:], color='#808080', alpha=0.75, linewidth=1.0, label='State A probability', linestyle='-')
 
        # Titles axes and labels at the aprropriate spots
        dx =  bin_size / 1e6
 
        if row == 0:
          ax.set_title(text_labels[col])
 
        ax.set_xlim((-dx, nx+dx))
        ax.set_ylim((0, 1.1*mv))
        ax.set_xlabel('Chr %s Position (Mb)' % chromo, fontsize=11)
        ax.axhline(0, -dx, nx+dx, color='#808080', alpha=0.5)
 
        if col == 0:
          ax.set_ylabel('$\Sigma[log_{10}(1+qual)]$')

        if row == 0:
          ax.legend(fontsize=11, frameon=False, ncol=3)
       
    file_obj.close()
    
    info('Output data written to %s' % file_path)

  if plot_graphs:
    plt.show()
  
  
if __name__ == '__main__':

  from argparse import ArgumentParser
  
  epilog = 'For further help email tstevens@mrc-lmb.cam.ac.uk or garethb@mrc-lmb.cam.ac.uk' 
  
  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='VCF_FILE', nargs='+', dest='i',
                         help='Input VCF format files containing variant calls for each parental genotype/strain. Files should be listed sequentially in pair order. Inputs may be gzipped (this is assumed by a .gz file extension).')

  arg_parse.add_argument('-b', default=DEFAULT_BIN_SIZE, type=int, metavar='BIN_SIZE',
                         help='Binned analysis region size (in bp) for defining HMM chromosome segments (). Default: %d bp.' % DEFAULT_BIN_SIZE)

  arg_parse.add_argument('-g', default=False, action='store_true',
                         help='Specifies that graphical output will be displayed for each VCF pair using matplotlib')

  arg_parse.add_argument('-m', default=DEFAULT_MIN_CHROMO_SIZE, type=float, metavar='MIN_CHROMO_SIZE',
                         help='Minimum length (in Mb) required for a contig/chromosome (as described in VCF header) to be analysed. Default: %.2f Mb.' % DEFAULT_MIN_CHROMO_SIZE)

  arg_parse.add_argument('-n', default=DEFAULT_NUM_ITER, type=int, metavar='NUM_HMM_ITER',
                         help='Number of iterations to perform when estimating Gaussian HMM probabilities using the Baum-Welch method. Default: %d' % DEFAULT_NUM_ITER)

  arg_parse.add_argument('-o', metavar='OUT_DIR',
                         help='Optional output directory for writing results. Defaults to the current working directory.')

  arg_parse.add_argument('-q', default=DEFAULT_QUALITY, type=float, metavar='',
                         help='Minimum quality (phred-scale) for accepting a SNP call, as described in the VCF data lines. Default: %.2f' % DEFAULT_QUALITY)
                         
  arg_parse.add_argument('-s', default=0, type=int, metavar='RANDOM_SEED',
                         help='Optional random seed value, i.e. to make repeat calculations deterministic.')
                         
  arg_parse.add_argument('-t', nargs='+', metavar='TEXT_LABEL',
                         help='Optional text labels to describe each input pair, which are used to name output files as {NAME}_%s.tsv, in the same order as the input VCF file pairs.' % FILE_TAG)
                         
  args = vars(arg_parse.parse_args(sys.argv[1:]))

  vcf_paths       = args['i']
  plot_graphs     = args['g']
  ran_seed        = args['s']
  out_dir         = args['o'] or './'
  text_labels     = args['t']
  bin_size        = args['b']
  min_qual        = args['q']
  min_chromo_size = int(args['m'] * 1e6)
  num_hmm_iter    = args['n']

  try:
    from matplotlib import pyplot as plt
  except ImportError:
    plot_graphs = False
  
  if ran_seed:
    np.random.seed(ran_seed)
  
  n_paths = len(vcf_paths)
  
  if n_paths < 2:
    fatal('At least two VCF file paths must be specified')
  
  if n_paths % 2 == 1:
    fatal('An even number of VCF paths (i.e. pairs of files) must be input. %d paths were specified' % n_paths)
  
  n_pairs = n_paths/2
  
  if text_labels:
    text_labels = list(text_labels)
  else:
    text_labels = []
  
  while len(text_labels) < n_pairs:
    label = 'pair_%d' % (1+len(text_labels))
    text_labels.append(label)
  
  if len(text_labels) > n_pairs:
    warn('Number of input text labels (%d) greater than the number of input pairs (%s)' % (len(text_labels), n_pairs))
    text_labels = text_labels[:n_pairs]
  
  abs_path = os.path.abspath(out_dir)
  if not os.path.exists(abs_path):
    fatal('Output directory "%d" does not exist')

  if not os.path.isdir(abs_path):
    fatal('Output path "%d" is not a directory')
  
  for vcf_path in vcf_paths:
    problem = check_file_problems(vcf_path)
  
    if problem:
      fatal(problem)
  
  vcf_paths_pairs = [(vcf_paths[i], vcf_paths[i+1]) for i in range(0, n_paths, 2)]
 
  train_hybrid_crossover_hmm(vcf_paths_pairs, text_labels, out_dir, bin_size,
                             min_qual, min_chromo_size, num_hmm_iter, plot_graphs)


  # Example ./hycco xgb3_vs12_clean.vcf.gz xgb3_vs13_clean.vcf.gz
