# R. Ruiz
# Feb 2025
# plot_EVAnlp_vs_xx_E2p5TeV_mu250GeV.py
# For details, see Bigaran & Ruiz [arXiv:2502.07878]
# 
# Summary: plots W PDF vs x at LLA, LP, and NLP (+ ratios)
# Usage: $ python ./plot_EVAnlp_vs_xx_E2p5TeV_mu250GeV.py
# output: evaNLP_Wpdf_vs_xx_Emu2p5TeV_mu250GeV.png and .pdf

# Structure:
# 1. define input classes
# 2. define functions (+ helper functions)
# 3. main program (plot) <-- change energy setup here
# 4. write to file

import math
from   matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from math import pi as pi

# #####################################
# 1.1 define SM and BSM input classes #
# #####################################

class smInputs:
    """SM inputs"""
    hbarc2 = 0.3893793721e9 # pb GeV^2
    #hbarc2 = 0.3893793721e12 # fb GeV^2
    MW = 80.41900 # GeV
    WW = 2.047600e+00 # GeV
    MZ = 91.1880e0 # GeV
    WZ = 2.441404e+00 # GeV
    MW2 = MW*MW
    MZ2 = MZ*MZ
    GF = 1.166390e-05 # GeV^-2 
    aS = 1.184000e-01
    NC = 3
    aEM = 1./1.325070e+02   # [derived input]
    sW2 = 1.0 - (MW2/MZ2)   # [derived input]
    cW2 = 1.0 - sW2         # [derived input]
    fourPiSq = 39.47841760435743 # 4pi**2 
    verbose=False 

# ######################
# 1.2 fetch SM inputs  #
# ######################

def get_eva_mufMin_byPID( vPID, fPID):
    return {
        7:  get_eva_mf_by_PID(fPID),
        22: get_eva_mf_by_PID(fPID),
        23: get_eva_mv_by_PID(vPID),
        24: get_eva_mv_by_PID(vPID)
    }.get(abs(vPID),-1)

def get_eva_mf_by_PID( fPID):
    # these must be the same as in ElectroweakFlux.inc
    return {
        1:  4.67e-3,
        2:  2.16e-3,
        3:  93.0e-3,
        4:  1.27e0,
        5:  4.18e0,
        6:  172.76e0,
        11: 0.5109989461e-3,
        13: 105.6583745e-3,
        15: 1.77686e0
    }.get(abs(fPID),-1)

def get_eva_mv_by_PID( vPID):
        return {
        7:      0e0,
        22:     0e0,
        23:     smInputs.MZ,
        24:     smInputs.MW
    }.get(abs(vPID),-1)    


# ####################################
# 2.1 PDFs for f_L -> f'_L splitting #
# ####################################

def get_eva_pdf_vp(gg2, gL2, muf, vPID, fPID, xx, ebeam, ievo=0, evaorder=0):
    #mufMin = get_eva_mufMin_byPID(vPID,fPID)
    if(evaorder < 0): 
        raise Exception("Invalid evaorder! evaorder = %s" % evaorder)
    elif(evaorder==2):  # next-to-leading power
        return calc_eva_fL_to_vp_nlp(gg2, gL2, muf, vPID, xx, ebeam, ievo)
    elif(evaorder==1):  # full leading power
        return calc_eva_fL_to_vp_xlp(gg2, gL2, muf, vPID, xx, ebeam, ievo)
    else:               # leading log approximation (default)
        return calc_eva_fL_to_vp_lla(gg2, gL2, muf, vPID, xx, ebeam, ievo)

# scale dependence of f_V- according to EVA accuracy
def get_eva_pdf_vm(gg2, gL2, muf, vPID, fPID, xx, ebeam, ievo=0, evaorder=0):
    #mufMin = get_eva_mufMin_byPID(vPID,fPID)
    if(evaorder < 0): 
        raise Exception("Invalid evaorder! evaorder = %s" % evaorder)
    elif(evaorder==2):  # next-to-leading power
        return calc_eva_fL_to_vm_nlp(gg2, gL2, muf, vPID, xx, ebeam, ievo)
    elif(evaorder==1):  # full leading power
        return calc_eva_fL_to_vm_xlp(gg2, gL2, muf, vPID, xx, ebeam, ievo)
    else:               # leading log approximation (default)
        return calc_eva_fL_to_vm_lla(gg2, gL2, muf, vPID, xx, ebeam, ievo)

# scale dependence of f_V0 according to EVA accuracy
def get_eva_pdf_v0(gg2, gL2, muf, vPID, fPID, xx, ebeam, ievo=0, evaorder=0):
    #mufMin = get_eva_mufMin_byPID(vPID,fPID)
    if(evaorder < 0): 
        raise Exception("Invalid evaorder! evaorder = %s" % evaorder)
    elif(evaorder==2):  # next-to-leading power
        return calc_eva_fL_to_v0_nlp(gg2, gL2, muf, vPID, xx, ebeam, ievo)
    elif(evaorder==1):  # full leading power
        return calc_eva_fL_to_v0_xlp(gg2, gL2, muf, vPID, xx, ebeam, ievo)
    else:               # leading log approximation (default)
        return calc_eva_fL_to_v0_lla(gg2, gL2, muf, vPID, xx, ebeam, ievo)
    
# f_V0 at lla
def calc_eva_fL_to_v0_lla(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        coup2 = gg2*gL2/smInputs.fourPiSq
        split = (1.0-xx) / (xx)
        oterm = 1.0
        xxlog = 1.0

        tmp = coup2 * split * oterm * xxlog
        return tmp

# f_V0 at LP
def calc_eva_fL_to_v0_xlp(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        coup2 = gg2*gL2/smInputs.fourPiSq
        split = (1.0-xx) / (xx)
        
        # O(1) term
        mu2 = muf*muf
        mv2 = (get_eva_mv_by_PID(vPID))**2
        muOmumv = 1.0 + mv2/mu2
        muOmumv = 1.0/muOmumv
        oterm = muOmumv

        xxlog = 1.0

        tmp = coup2 * split * oterm * xxlog
        return tmp

# f_V0 at nlp
def calc_eva_fL_to_v0_nlp(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        mv2 = (get_eva_mv_by_PID(vPID))**2
        ev2 = (xx*ebeam)**2
        mvOev = mv2 / ev2 / 2.0 
        
        # xlp terms
        f0xlp = calc_eva_fL_to_v0_xlp(gg2, gL2,muf, vPID, xx, ebeam, ievo)
        fpxlp = calc_eva_fL_to_vp_xlp(gg2, gL2,muf, vPID, xx, ebeam, ievo)
        fmxlp = calc_eva_fL_to_vm_xlp(gg2, gL2,muf, vPID, xx, ebeam, ievo)
        
        # combine
        tmp = f0xlp - mvOev*(fpxlp+fmxlp)
        return tmp

# f_V+ at lla
def calc_eva_fL_to_vp_lla(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        coup2 = gg2*gL2/smInputs.fourPiSq
        split = (1.0-xx)**2 / (2.0*xx)
        oterm = 1.0

        # = log(muf2 / mv2)
        mu2 = muf*muf
        mv2 = (get_eva_mv_by_PID(vPID))**2
        muOmv = mu2/mv2
        xxlog = math.log(muOmv)

        tmp =  coup2 * split * oterm * xxlog
        return tmp

# f_V+ at xlp
def calc_eva_fL_to_vp_xlp(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        coup2 = gg2*gL2/smInputs.fourPiSq
        split = (1.0-xx)**2 / (2.0*xx)

        # = log(muf2 / mv2)
        mu2 = muf*muf
        mv2 = (get_eva_mv_by_PID(vPID))**2
        mumvOmv = mu2/mv2 + 1.0
        xxlog = math.log(mumvOmv)

        # O(1) term
        muOmumv = 1.0 + mv2/mu2
        muOmumv = 1.0/muOmumv
        oterm = muOmumv

        tmp =  coup2 * split * (xxlog - oterm)
        return tmp

# f_V+ at nlp
def calc_eva_fL_to_vp_nlp(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        # = fV+^nlp * (4pi^2 z / g^2 gL^2)
        mu2 = muf*muf
        mv2 = (get_eva_mv_by_PID(vPID))**2
        ev2 = (xx*ebeam)**2
        
        # ratios
        mvOev   = mv2/ev2
        muOev   = mu2 / ev2 / 4.0
        xxrat   = (2.0-xx)/(1.0-xx)

        # xlp terms
        fpxlp = calc_eva_fL_to_vp_xlp(gg2, gL2,muf, vPID, xx, ebeam, ievo)
        f0xlp = calc_eva_fL_to_v0_xlp(gg2, gL2,muf, vPID, xx, ebeam, ievo)

        # combine
        fxlpTerm = (1.0 + xxrat*mvOev)*fpxlp
        flipTerm = (2.0-xx)*muOev*f0xlp

        tmp = fxlpTerm - flipTerm
        return tmp

# f_V+ at lla
def calc_eva_fL_to_vm_lla(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        coup2 = gg2*gL2/smInputs.fourPiSq
        split = (1.0) / (2.0*xx)
        oterm = 1.0

        # = log(muf2 / mv2)
        mu2 = muf*muf
        mv2 = (get_eva_mv_by_PID(vPID))**2
        muOmv = mu2/mv2
        xxlog = math.log(muOmv)

        tmp =  coup2 * split * oterm * xxlog
        return tmp

# f_V+ at xlp
def calc_eva_fL_to_vm_xlp(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        coup2 = gg2*gL2/smInputs.fourPiSq
        split = (1.0) / (2.0*xx)

        # = log(muf2 / mv2)
        mu2 = muf*muf
        mv2 = (get_eva_mv_by_PID(vPID))**2
        mumvOmv = mu2/mv2 + 1.0
        xxlog = math.log(mumvOmv)

        # O(1) term
        muOmumv = 1.0 + mv2/mu2
        muOmumv = 1.0/muOmumv
        oterm = muOmumv

        tmp =  coup2 * split * (xxlog - oterm)
        return tmp

# f_V- at nlp
def calc_eva_fL_to_vm_nlp(gg2, gL2, muf, vPID, xx, ebeam, ievo=0):
        mu2 = muf*muf
        mv2 = (get_eva_mv_by_PID(vPID))**2
        ev2 = (xx*ebeam)**2
        
        # ratios
        mvOev   = mv2/ev2
        muOev   = mu2 / ev2 / 4.0
        xxrat   = (2.0-xx)/(1.0-xx)

        # xlp terms
        fmxlp = calc_eva_fL_to_vm_xlp(gg2, gL2,muf, vPID, xx, ebeam, ievo)
        f0xlp = calc_eva_fL_to_v0_xlp(gg2, gL2,muf, vPID, xx, ebeam, ievo)

        # combine
        fxlmTerm = (1.0 + (2.0-xx)*mvOev)*fmxlp
        flipTerm = xxrat*muOev*f0xlp

        tmp = fxlmTerm - flipTerm
        return tmp

# ####################################
# 2.2 PDFs for f_R -> f'_R splitting #
# ####################################

def calc_eva_fR_to_v0_lla(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_v0_lla(gg2, gR2, muf, vPID, xx, ebeam, ievo)

def calc_eva_fR_to_v0_xlp(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_v0_xlp(gg2, gR2, muf, vPID, xx, ebeam, ievo)

def calc_eva_fR_to_v0_nlp(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_v0_nlp(gg2, gR2, muf, vPID, xx, ebeam, ievo)

def calc_eva_fR_to_vp_lla(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_vm_lla(gg2, gR2, muf, vPID, xx, ebeam, ievo)

def calc_eva_fR_to_vp_xlp(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_vm_xlp(gg2, gR2, muf, vPID, xx, ebeam, ievo)

def calc_eva_fR_to_vp_nlp(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_vm_nlp(gg2, gR2, muf, vPID, xx, ebeam, ievo)

def calc_eva_fR_to_vm_lla(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_vp_lla(gg2, gR2, muf, vPID, xx, ebeam, ievo)

def calc_eva_fR_to_vm_xlp(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_vp_xlp(gg2, gR2, muf, vPID, xx, ebeam, ievo)

def calc_eva_fr_to_vm_nlp(gg2, gR2, muf, vPID, xx, ebeam, ievo=0):
     return calc_eva_fL_to_vp_nlp(gg2, gR2, muf, vPID, xx, ebeam, ievo)

# ################################
# 2.3 wrapper functions for main #
# ################################

def get_eva_W0_by_order(muf, xx, ebeam, ievo=0, evaorder=0):
    vPID = 24
    fPID = 13 # muon
    gg2 = (0.5)**2
    gL2 = (1.0)**2

    tmp = get_eva_pdf_v0(gg2, gL2, muf, vPID, fPID, xx, ebeam, ievo, evaorder)
    return tmp

def get_eva_Wp_by_order(muf, xx, ebeam, ievo=0, evaorder=0):
    vPID = 24
    fPID = 13 # muon
    gg2 = (0.5)**2
    gL2 = (1.0)**2

    tmp = get_eva_pdf_vp(gg2, gL2, muf, vPID, fPID, xx, ebeam, ievo, evaorder)
    return tmp

def get_eva_Wm_by_order(muf, xx, ebeam, ievo=0, evaorder=0):
    vPID = 24
    fPID = 13 # muon
    gg2 = (0.5)**2
    gL2 = (1.0)**2

    tmp = get_eva_pdf_vm(gg2, gL2, muf, vPID, fPID, xx, ebeam, ievo, evaorder)
    return tmp

def lighten_color(color, amount=0.5):
    # https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])    


# ########################
# 3. main program / plot #
# ########################

# generate (x,y) values
nrPoints = 1000
scale = 250
ebeam = 2500
xPID = 24
xMin = get_eva_mv_by_PID(xPID) / ebeam
x800 = 800 / ebeam
xMuf = scale / ebeam
xVal = list(np.logspace(-2.1,-1e-6,nrPoints)) # from 0.001 to 0.99977

print(f"computing W boson PDFs for {ebeam=} TeV, {scale=} GeV")
print(f"minimum x thresholds are at {xMin=}\t{xMuf=}")

yValEVA_W0_lla = list(get_eva_W0_by_order(scale, xx, ebeam,0,0) for xx in xVal)
yValEVA_Wp_lla = list(get_eva_Wp_by_order(scale, xx, ebeam,0,0) for xx in xVal)
yValEVA_Wm_lla = list(get_eva_Wm_by_order(scale, xx, ebeam,0,0) for xx in xVal)

yValEVA_W0_xlp = list(get_eva_W0_by_order(scale, xx, ebeam,0,1) for xx in xVal)
yValEVA_Wp_xlp = list(get_eva_Wp_by_order(scale, xx, ebeam,0,1) for xx in xVal)
yValEVA_Wm_xlp = list(get_eva_Wm_by_order(scale, xx, ebeam,0,1) for xx in xVal)

yValEVA_W0_nlp = list(get_eva_W0_by_order(scale, xx, ebeam,0,2) for xx in xVal)
yValEVA_Wp_nlp = list(get_eva_Wp_by_order(scale, xx, ebeam,0,2) for xx in xVal)
yValEVA_Wm_nlp = list(get_eva_Wm_by_order(scale, xx, ebeam,0,2) for xx in xVal)

yValEVA_W0_nlpOlla = list(yValEVA_W0_nlp[kk]/yValEVA_W0_lla[kk] for kk in range(0,len(xVal)))
yValEVA_Wp_nlpOlla = list(yValEVA_Wp_nlp[kk]/yValEVA_Wp_lla[kk] for kk in range(0,len(xVal)))
yValEVA_Wm_nlpOlla = list(yValEVA_Wm_nlp[kk]/yValEVA_Wm_lla[kk] for kk in range(0,len(xVal)))

yValEVA_W0_nlpOxlp = list(yValEVA_W0_nlp[kk]/yValEVA_W0_xlp[kk] for kk in range(0,len(xVal)))
yValEVA_Wp_nlpOxlp = list(yValEVA_Wp_nlp[kk]/yValEVA_Wp_xlp[kk] for kk in range(0,len(xVal)))
yValEVA_Wm_nlpOxlp = list(yValEVA_Wm_nlp[kk]/yValEVA_Wm_xlp[kk] for kk in range(0,len(xVal)))

if(smInputs.verbose):
    for kk in range(0,len(xVal)):
        print(f"{xVal[kk]=}")
        tmpW0 = (yValEVA_W0_xlp[kk]/yValEVA_W0_lla[kk] - 1.)*100.
        tmpWp = (yValEVA_Wp_xlp[kk]/yValEVA_Wp_lla[kk] - 1.)*100
        tmpWm = (yValEVA_Wm_xlp[kk]/yValEVA_Wm_lla[kk] - 1.)*100
        print("difference between LP and LLA [%%] for W0, W+, W- : (%f,%f,%f)" % (tmpW0,tmpWp,tmpWm))

# ########
# colors #
# ########
colorPink=(245/255, 169/255, 184/255) # RGB
colorTeal=(91/255, 206/255, 250/255) # RGB

# #####################
# aesthetics (global) #
# #####################
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif" : ['Computer Modern Roman']
})
fontUpper = {'size':17}
fontLower = {'size':15}
boxProp = dict(boxstyle='square', facecolor='white', alpha=0.5)

# ######
# plot #
# ######
fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0, height_ratios=[3, 1, 1])
ax = gs.subplots(sharex=True, sharey=False)

w0lla, = ax[0].plot(xVal,yValEVA_W0_lla, label=r'LLA', color="black", linestyle="solid")
wplla, = ax[0].plot(xVal,yValEVA_Wp_lla, label=r'LLA', color=lighten_color(colorPink,2),  linestyle="solid")
wmlla, = ax[0].plot(xVal,yValEVA_Wm_lla, label=r'LLA', color=lighten_color(colorTeal,2), linestyle="solid")

w0nlp, = ax[0].plot(xVal,yValEVA_W0_nlp, label=r'NLP', color="black", linestyle="dashdot")
wpnlp, = ax[0].plot(xVal,yValEVA_Wp_nlp, label=r'NLP', color=lighten_color(colorPink,2),  linestyle="dashdot")
wmnlp, = ax[0].plot(xVal,yValEVA_Wm_nlp, label=r'NLP', color=lighten_color(colorTeal,2), linestyle="dashdot")

ax[0].axvspan(0,xMin,linestyle="solid",color="gray",alpha=.4)
ax[0].axvspan(0,xMuf,linestyle="solid",color="gray",alpha=.2)
ax[0].legend(handles=[w0lla,w0nlp],frameon=False,loc='upper right', bbox_to_anchor=(0.95, 0.75))

# first subplot
ax[1].axhline(1.0,linestyle="solid",color="black",alpha=.5)
ax[1].axhline(0.8,linestyle="dotted",color="black",alpha=.5)
ax[1].plot(xVal,yValEVA_W0_nlpOlla, label=r'NLP / LLA', color="black", linestyle="dashdot")
ax[1].plot(xVal,yValEVA_Wp_nlpOlla, label=r'NLP / LLA', color=lighten_color(colorPink,2),  linestyle="dashdot")
ax[1].plot(xVal,yValEVA_Wm_nlpOlla, label=r'NLP / LLA', color=lighten_color(colorTeal,2), linestyle="dashdot")
ax[1].axvspan(0,xMin,linestyle="solid",color="gray",alpha=.4)
ax[1].axvspan(0,xMuf,linestyle="solid",color="gray",alpha=.2)

# second subplot
ax[2].axhline(1.0,linestyle="solid",color="black",alpha=.5)
ax[2].axhline(0.8,linestyle="dotted",color="black",alpha=.5)
ax[2].plot(xVal,yValEVA_W0_nlpOxlp, label=r'NLP / LP', color="black", linestyle="dashed")
ax[2].plot(xVal,yValEVA_Wp_nlpOxlp, label=r'NLP / LP', color=lighten_color(colorPink,2),  linestyle="dashed")
ax[2].plot(xVal,yValEVA_Wm_nlpOxlp, label=r'NLP / LP', color=lighten_color(colorTeal,2), linestyle="dashed")
ax[2].axvspan(0,xMin,linestyle="solid",color="gray",alpha=.4)
ax[2].axvspan(0,xMuf,linestyle="solid",color="gray",alpha=.2)

# aesthetics (x-axis)
plt.xlabel(r'momentum fraction, $z\ =\ E_V\ /\ E_\mu$',fontsize=17)
plt.xlim(0.9e-2,1.0) # set range of x-axis
plt.xscale('log')
plt.xticks(fontsize=11)

for axis in ax:
    axis.tick_params(which="major",top=True,bottom=True,left=True,right=True,direction="in")
    axis.tick_params(which="minor",top=True,bottom=True,left=True,right=True,direction="in")
    axis.minorticks_on()

# aesthetics (y-axis)
ax[0].set_ylabel(r"$f(z,\mu_f)$",fontdict=fontUpper)
ax[0].set(ylim=(9e-4,1.5),yscale='log')

ax[1].set_ylabel(r'ratio',fontdict=fontLower)
ax[1].set(ylim=(0.0,1.25))

ax[2].set_ylabel(r'ratio',fontdict=fontLower)
ax[2].set(ylim=(0.0,1.25))

# aesthetics (inserts)
ax[0].text(4.25e-2,0.65, r"$\mu^- \to W^-_{\lambda} \nu_\mu$",fontsize=15)
ax[0].text(0.275,0.65, r"$E_{\mu} = 2.5$ TeV",fontsize=15)
ax[0].text(0.295,0.325, r"$\mu_f = 250$ GeV",fontsize=13)
ax[0].text(1.125e-2,1.5e-3, r"$E_V < M_W$",fontsize=13,bbox=boxProp)
ax[0].text(4.25e-2,1.5e-3, r"$E_V < \mu_f$",fontsize=13,bbox=boxProp)
ax[0].text(1.70e-2,0.01,r"$W_{\lambda=0}$", color=ax[0].get_lines()[0].get_color(), fontsize=15)
ax[0].text(0.275,2e-3,r"$W_{\lambda=+}$", color=ax[0].get_lines()[1].get_color(), fontsize=15)
ax[0].text(0.500,1.8e-2,r"$W_{\lambda=-}$", color=ax[0].get_lines()[2].get_color(), fontsize=15)

ax[1].text(0.375,0.25,r"NLP / LLA",fontsize=13)
ax[1].text(0.020,0.25,r"$W_{\lambda=0}$", color=ax[0].get_lines()[0].get_color(), fontsize=13)
ax[1].text(0.125,0.20,r"$W_{\lambda=+}$", color=ax[0].get_lines()[1].get_color(), fontsize=13)
ax[1].text(0.225,0.25,r"$W_{\lambda=-}$", color=ax[0].get_lines()[2].get_color(), fontsize=13)

ax[2].text(0.375,0.65,r"NLP / LP",fontsize=13)
ax[2].text(0.02,0.25,r"$W_{\lambda=0}$", color=ax[0].get_lines()[0].get_color(), fontsize=13)
ax[2].text(0.11,0.25,r"$W_{\lambda=+}$", color=ax[0].get_lines()[1].get_color(), fontsize=13)
ax[2].text(0.045,0.25,r"$W_{\lambda=-}$", color=ax[0].get_lines()[2].get_color(), fontsize=13)

# ###########################
# 4. output / write to file #
# ###########################
outDPI = 450
outTitle = "evaNLP_Wpdf_vs_xx_Emu2p5TeV_mu250GeV"
print("generating "+outTitle+".png at dpi= %i" % outDPI)
plt.savefig(outTitle+'.png',bbox_inches='tight',dpi=outDPI)
print("generating "+outTitle+".pdf")
plt.savefig(outTitle+'.pdf',bbox_inches='tight',dpi=450)

print("done! have a nice day.")

