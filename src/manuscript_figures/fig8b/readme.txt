Instructions for generating Fig 8b

--------------------------------------------------------------------------------

1. Compile the code "gap": type "make".

2. Run the code

gap < ei.txt
mv out.convert ei.match
gap < sc.txt
mv out.convert sc.match

The input files (ei.txt, sc.txt) are generated from the fig7 KDE cache
(kde_PC1_*.csv and kde_weightedStim_*.csv) and have the following format:

column 1: z-score
column 2: prior-induced gap
column 3: slope of psychometric curve

3. Fig 8b

The output files ei.match and sc.match have the following format:

column 1: z-score (copied from input file)
column 2: sigma
column 3: pr^hat

Fig 8b is a plot of pr^hat versus 1/(sqrt(2*pi) * sigma).
