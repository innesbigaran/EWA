# for processing and plotting Invariant mass distributions from LHE files from MG5. 
# Innes Bigaran. Last updated Feb 13 2025 

import itertools
import pylhe
import awkward as ak
import numpy as np
import hist


def get_events(filename):
    mg5events = pylhe.read_lhe_with_attributes(filename)
    # Create akward array objects for convenience
    mg5events = pylhe.to_awkward(mg5events)

    return mg5events

def contract_four_vectors(vec1, vec2):
    temp = sum([a*b for a,b in zip(vec1[1:],vec2[1:])])
    temp = vec1[0]*vec2[0] - temp
    return (temp)
	
def ArrayRatio(array1, array2):
    lengt=len(array1)
    arrayout=np.zeros(lengt)
    for i in range(lengt):
        if array2[i]==0:
                arrayout[i]=0.
        else:
            arrayout[i]=array1[i]/array2[i]
    return arrayout
	
def InvMass3a(eventsME):

    Gammas=(eventsME['particles','id'] == 22)
    
    p_A1=np.array([np.array([mom[0]['t'], mom[0]['x'], mom[0]['y'], mom[0]['z']]) for mom in eventsME["particles", "vector"][Gammas]])
    p_A2=np.array([np.array([mom[1]['t'], mom[1]['x'], mom[1]['y'], mom[1]['z']]) for mom in eventsME["particles", "vector"][Gammas]])
    p_A3=np.array([np.array([mom[2]['t'], mom[2]['x'], mom[2]['y'], mom[2]['z']]) for mom in eventsME["particles", "vector"][Gammas]])
    temp = p_A1 + p_A2 + p_A3
    m_3a = [np.sqrt(contract_four_vectors(entry,entry)) for entry in temp]
    
    return [m_3a]

def InvMass2h(eventsME):
    
    Higgs=(eventsME['particles','id'] == 25)
    
    p_Higgs1=np.array([np.array([mom[0]['t'], mom[0]['x'], mom[0]['y'], mom[0]['z']]) for mom in eventsME["particles", "vector"][Higgs]])
    p_Higgs2=np.array([np.array([mom[1]['t'], mom[1]['x'], mom[1]['y'], mom[1]['z']]) for mom in eventsME["particles", "vector"][Higgs]])
    temp = p_Higgs1 + p_Higgs2 
    m_2H = [np.sqrt(contract_four_vectors(entry,entry)) for entry in temp]
    
    return [m_2H]

def InvMass2a(eventsME):
    
    Gammas=(eventsME['particles','id'] == 22)
    
    p_A1=np.array([np.array([mom[0]['t'], mom[0]['x'], mom[0]['y'], mom[0]['z']]) for mom in eventsME["particles", "vector"][Gammas]])
    p_A2=np.array([np.array([mom[1]['t'], mom[1]['x'], mom[1]['y'], mom[1]['z']]) for mom in eventsME["particles", "vector"][Gammas]])
    temp = p_A1 + p_A2 
    m_2A = [np.sqrt(contract_four_vectors(entry,entry)) for entry in temp]
    
    return [m_2A]


def processfile3a(filename):
	events=get_events(filename)
	Nevents=pylhe.read_num_events(filename)
	xsec=pylhe.read_lhe_init(filename)['procInfo'][0]['xSection']
	weight=events["eventinfo", "weight"][0]
	print("Number of events:", Nevents)
	print("Cross Section (pb):", xsec)
	print("Weight (pb):", weight)
	InvMass=InvMass3a(events)
	#weights=events["eventinfo", "weight"]/Nevents
	Luminosity= Nevents/(xsec*1000*1000)
	return {"InvMass":InvMass,"Luminosity":Luminosity, 'xSec':xsec, 'weight':events["eventinfo", "weight"]}
	
def processfile2h(filename):
	events=get_events(filename)
	Nevents=pylhe.read_num_events(filename)
	xsec=pylhe.read_lhe_init(filename)['procInfo'][0]['xSection']
	weight=events["eventinfo", "weight"][0]
	print("Number of events:", Nevents)
	print("Cross Section (pb):", xsec)
	print("Weight (pb):", weight)
	InvMass=InvMass2h(events)
	#weights=events["eventinfo", "weight"]/Nevents
	Luminosity= Nevents/(xsec*1000*1000)
	return {"InvMass":InvMass,"Luminosity":Luminosity, 'xSec':xsec, 'weight':events["eventinfo", "weight"]}

def processfile2a(filename):
	events=get_events(filename)
	Nevents=pylhe.read_num_events(filename)
	xsec=pylhe.read_lhe_init(filename)['procInfo'][0]['xSection']
	weight=np.abs(events["eventinfo", "weight"][0])
	print("Number of events:", Nevents)
	print("Cross Section (pb):", xsec)
	print("Weight (pb):", weight)
	InvMass=InvMass2a(events)
	Luminosity= Nevents/(xsec*1000*1000)
	return {"InvMass":InvMass,"Luminosity":Luminosity, 'xSec':xsec, 'weight':events["eventinfo", "weight"]}


def FillHistogram(Dict):
	#takes as input the output of a file process above 
	#output plot as ax.hist(output[0], output[1], weights=output[3])
	bins = np.linspace(300, 3000, 30) #min, max, number of bins.
	hist, bin_edges=np.histogram(Dict["InvMass"], bins=bins,  weights=[Dict["weight"]]) 
	bin_widths = np.diff(bin_edges)
	bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
	cross_sections = 1e6*hist/(bin_widths)
	area = np.sum(cross_sections * bin_widths)
	print(area)
	return [bin_midpoints, bin_midpoints[:-1], cross_sections, bin_edges]

	
