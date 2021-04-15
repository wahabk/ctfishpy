import ctfishpy

ctreader = ctfishpy.CTreader()
wahab_samples 	= [40,78,200,218,240,277,330,337,341,462,464,364,385]

mariel_samples	= [256,421,423,242,463,259,459]
zac_samples		= [582,257,443,461] 

for num in mariel_samples:
	ct, metadata = ctreader.read(num, align=True)

	align=False
	if num in [40,78,200,218,240,277,330,337,341,462,464,364,385]: align = True
	label = ctreader.read_label('Otoliths', num, align=align, is_amira=True)
	
	ctreader.view(ct, label=label)