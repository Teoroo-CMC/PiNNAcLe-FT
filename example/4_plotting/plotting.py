import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

EXAMPLE_DIR = Path("{your_PiNNAcLe_FT_Path}/example")
WORK_DIR = EXAMPLE_DIR / "4_plotting"
FT_DIR = EXAMPLE_DIR / "3_pinet2_ft"

# for matplotlib
listStyles = ['-','--','-.',':']
listMarkers = ['o','s','<','>','p','v','h','^','d',',','x','+',]
# https://zhuanlan.zhihu.com/p/508870810, '#a9a9a9'
listDistColors = ['#a9a9a9', '#f600e6', '#00ff00', '#0080ff', '#ff8000', '#4b00b0', '#000000', '#ae0132', '#0c811d']

# global text font Arial/Helvetica: from https://www.nature.com/ncomms/submit/how-to-submit
# also see https://www.nature.com/nature/for-authors/initial-submission
plt.rcdefaults() # reset
plt.rc('font', family='sans-serif', size=10)
plt.rcParams['font.sans-serif'] = 'Arial'

# global direction of axes
plt.rc('xtick', direction='out')
plt.rc('ytick', direction='out')

# 
def drawFinetuningCurve(nSeed):

    import re
    from glob import glob
    import matplotlib.patheffects as pe

    initial_training_steps = 5000000

    strTargetSystem = "wat64_h3o+"
    bRMSEOverSystem = False # pinnacle can be used on more than one systems, here is the control for the calculation of RMSE
    pinnacle_output_dir = FT_DIR / f"output-cp2k-from-model-seed{nSeed}"

    # get validation RMSE results
    listPiNNLogFile = glob(f'{pinnacle_output_dir}/models/gen*/model1/pinn.log')
    listPiNNLogGen = [int(re.search('gen(\d+)', l)[1]) for l in listPiNNLogFile]
    listValidRMSE = [np.loadtxt(l)[-1] for l in listPiNNLogFile] # the last line
    arrPiNNRMSE = np.array([[g,*e] for g, e in zip(listPiNNLogGen, listValidRMSE)])
    arrPiNNRMSE = arrPiNNRMSE[arrPiNNRMSE[:,0].argsort()]

    # get validation max results
    listValidMAX = []
    listPiNNAppendFile = glob(f'{pinnacle_output_dir}/models/gen*/model1/append.log')
    listPiNNAppendGen = [int(re.search('gen(\d+)', l)[1]) for l in listPiNNAppendFile]
    for strFile in listPiNNAppendFile:
        with open(strFile, "r") as f:
            listLines = f.readlines()
            assert "energy and force measures on validation set" in listLines[1]
            dEnerValidMax = float(re.search(r"emax=([0-9]+(?:\.[0-9]+)?)", listLines[1])[1])
            dForceValidMax = float(re.search(r"fmax=([0-9]+(?:\.[0-9]+)?)", listLines[1])[1])
            listValidMAX.append([dEnerValidMax, dForceValidMax])
    arrPiNNMAX = np.array([[g,*e] for g, e in zip(listPiNNAppendGen, listValidMAX)])
    arrPiNNMAX = arrPiNNMAX[arrPiNNMAX[:,0].argsort()]

    # get md simulation time scale
    listTrajFile = glob(f'{pinnacle_output_dir}/md/gen*/{strTargetSystem}/asemd.log')
    listTrajGen = [int(re.search('gen(\d+)', l)[1]) for l in listTrajFile]
    listMDScale = [np.loadtxt(l, skiprows=1)[-1,0] for l in listTrajFile]
    arrMDScale = np.array([*zip(listTrajGen, listMDScale)])
    arrMDScale = arrMDScale[arrMDScale[:,0].argsort()]

    # get the test error
    listTestGen = []
    dictSys2EnerMax = {}
    dictSys2EnerNum = {}
    dictSys2EnerSSE = {}
    dictSys2ForceMax = {}
    dictSys2ForceNum = {}
    dictSys2ForceSSE = {}
    listAllSystems = []
    e_max_tolerance, e_rmse_tolerance, f_max_tolerance, f_rmse_tolerance  = 999.0, 999.0, 999.0, 999.0
    logs = pinnacle_output_dir / "pinnacle.log"
    with open(logs, "r") as f:
        listLines = f.readlines()
        for strLine in listLines:
            if "Convergence tolerance" in strLine:
                e_max_tolerance = float(re.search(r"emaxtol=([0-9]+(?:\.[0-9]+)?);", strLine)[1])*1e3
                e_rmse_tolerance = float(re.search(r"ermsetol=([0-9]+(?:\.[0-9]+)?);", strLine)[1])*1e3
                f_max_tolerance = float(re.search(r"fmaxtol=([0-9]+(?:\.[0-9]+)?);", strLine)[1])*1e3
                f_rmse_tolerance = float(re.search(r"frmsetol=([0-9]+(?:\.[0-9]+)?).", strLine)[1])*1e3
                continue
            elif "esse=" not in strLine:
                continue
            
            listItem = strLine.strip("\n").split()
            nGenIndex = int(listItem[0].split("/")[0].strip("[gen"))
            if nGenIndex not in listTestGen:
                listTestGen.append(nGenIndex)

            strCurSystem = listItem[0].split("/")[1].strip("]")
            if strCurSystem not in listAllSystems:
                listAllSystems.append(strCurSystem)

            if strCurSystem not in dictSys2EnerNum.keys():
                dictSys2EnerMax[strCurSystem] = [float(re.findall(r"max=([0-9]+(?:\.[0-9]+)?)", strLine)[0])]
                dictSys2EnerNum[strCurSystem] = [float(re.search(r"enum=([0-9]+(?:\.[0-9]+)?)", strLine)[1])]
                dictSys2EnerSSE[strCurSystem] = [float(re.search(r"esse=([0-9]+(?:\.[0-9]+)?)", strLine)[1])]
                dictSys2ForceMax[strCurSystem] = [float(re.findall(r"max=([0-9]+(?:\.[0-9]+)?)", strLine)[1])]
                dictSys2ForceNum[strCurSystem] = [float(re.search(r"fnum=([0-9]+(?:\.[0-9]+)?)", strLine)[1])]
                dictSys2ForceSSE[strCurSystem] = [float(re.search(r"fsse=([0-9]+(?:\.[0-9]+)?)", strLine)[1])]
            else:
                dictSys2EnerMax[strCurSystem].append(float(re.findall(r"max=([0-9]+(?:\.[0-9]+)?)", strLine)[0]))
                dictSys2EnerNum[strCurSystem].append(float(re.search(r"enum=([0-9]+(?:\.[0-9]+)?)", strLine)[1]))
                dictSys2EnerSSE[strCurSystem].append(float(re.search(r"esse=([0-9]+(?:\.[0-9]+)?)", strLine)[1]))
                dictSys2ForceMax[strCurSystem].append(float(re.findall(r"max=([0-9]+(?:\.[0-9]+)?)", strLine)[1]))
                dictSys2ForceNum[strCurSystem].append(float(re.search(r"fnum=([0-9]+(?:\.[0-9]+)?)", strLine)[1]))
                dictSys2ForceSSE[strCurSystem].append(float(re.search(r"fsse=([0-9]+(?:\.[0-9]+)?)", strLine)[1])) 

    # check the length of test results
    for strCurSystem in listAllSystems:
        assert len(listTestGen) == len(dictSys2EnerMax[strCurSystem])
        assert len(listTestGen) == len(dictSys2EnerNum[strCurSystem])
        assert len(listTestGen) == len(dictSys2EnerSSE[strCurSystem])
        assert len(listTestGen) == len(dictSys2ForceMax[strCurSystem])
        assert len(listTestGen) == len(dictSys2ForceNum[strCurSystem])
        assert len(listTestGen) == len(dictSys2ForceSSE[strCurSystem])

    # calculate the final RMSE
    strCurSystem = ""
    e_max, e_rmse, f_max, f_rmse = [], [], [], []
    for nGenIndex in range(0, len(listTestGen)):
        e_max_gen = 0.0
        e_num = 0.0
        e_value = 0.0
        f_max_gen = 0.0
        f_num = 0.0
        f_value = 0.0
        for strCurSystem in listAllSystems:
            if (not bRMSEOverSystem) and (strCurSystem != strTargetSystem):
                continue
            e_max_gen = max(e_max_gen, dictSys2EnerMax[strCurSystem][nGenIndex])
            e_num += dictSys2EnerNum[strCurSystem][nGenIndex]
            e_value += dictSys2EnerSSE[strCurSystem][nGenIndex]
            f_max_gen = max(f_max_gen, dictSys2ForceMax[strCurSystem][nGenIndex])
            f_num += dictSys2ForceNum[strCurSystem][nGenIndex]
            f_value += dictSys2ForceSSE[strCurSystem][nGenIndex]
        e_max.append(e_max_gen*1e3)
        e_rmse.append(np.sqrt(e_value/e_num)*1e3)
        f_max.append(f_max_gen*1e3)
        f_rmse.append(np.sqrt(f_value/f_num)*1e3)

    # do the plot
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=[7, 7], 
                                    sharex=True, 
                                    gridspec_kw={'hspace':0})
    tax = ax1.twinx()
    l1, = ax1.plot(arrPiNNRMSE[:,0], (arrPiNNRMSE[:,1]-initial_training_steps)/1e3, 'k.-')
    l2, = tax.plot(arrMDScale[:,0], arrMDScale[:,1], '.-', color='tab:red')
    l3, = ax2.plot(arrPiNNRMSE[:,0], arrPiNNRMSE[:,2]*1e3, '.-')
    l4, = ax2.plot(listTestGen, e_rmse, '.-')
    l5, = ax3.plot(arrPiNNRMSE[:,0], arrPiNNRMSE[:,4]*1e3, '.-')
    l6, = ax3.plot(listTestGen, f_rmse, '.-')
    l7, = ax4.plot(arrPiNNMAX[:,0], arrPiNNMAX[:,1], '.-')
    l8, = ax4.plot(listTestGen, e_max, '.-')
    l9, = ax5.plot(arrPiNNMAX[:,0], arrPiNNMAX[:,2], '.-')
    l10, = ax5.plot(listTestGen, f_max, '.-')

    # threshold
    ax2.plot([0,listTestGen[-1]], [e_rmse_tolerance,e_rmse_tolerance], 'k--', lw=1)
    ax3.plot([0,listTestGen[-1]], [f_rmse_tolerance,f_rmse_tolerance], 'k--', lw=1)
    ax4.plot([0,listTestGen[-1]], [e_max_tolerance,e_max_tolerance], 'k--', lw=1)
    ax5.plot([0,listTestGen[-1]], [f_max_tolerance,f_max_tolerance], 'k--', lw=1)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()

    tax.set_yscale('log')
    ax1.set_ylabel('Training Steps\n [Thousand]')
    ax2.set_ylabel('E RMSE\n [meV/atom]')
    ax3.set_ylabel('F RMSE\n [meV/$\AA$]')
    ax4.set_ylabel('E MAX\n [meV/atom]')
    ax5.set_ylabel('F MAX\n [meV/$\AA$]')
    ax5.set_xlabel('Generation')
    tax.tick_params(axis='y', labelcolor='tab:red')
    tax.set_ylabel('Sampling Time [ps]', color='tab:red')
    # ax1.set_yticks(np.arange(0,11,2))
    # ax2.set_yticks(np.arange(0,10,2))
    # ax3.set_yticks(np.arange(10,150,20))
    # tax.set_yticks([1e0, 1e1, 1e2, 1e3])
    ax5.set_xticks(np.arange(0,np.max(listTestGen)+1,2))
    # ax1.set_xlim(-2,42)
    # ax1.set_ylim(0.1,9.9)
    ax2.set_ylim(0,4.9)
    ax3.set_ylim(20,99)
    ax4.set_ylim(0,9)
    ax5.set_ylim(20,699)
    # tax.set_ylim(2e-1,2e3)

    leg1 = ax1.legend([l1, l2], ['Steps', 'Time'], loc=4, frameon=True)
    leg2 = ax2.legend([l3, l4], ['Eval', 'Test'],  loc=1, frameon=True)
    leg3 = ax3.legend([l5, l6], ['Eval', 'Test'],  loc=1, frameon=True)
    leg4 = ax4.legend([l7, l8], ['Eval', 'Test'],  loc=1, frameon=True)
    leg5 = ax5.legend([l9, l10], ['Eval', 'Test'],  loc=1, frameon=True)

    mype = [pe.withStroke(linewidth=4, foreground='w')]
    for leg in [leg1, leg2, leg3, leg4, leg5]:
        for t in leg.get_texts():
            t.set_path_effects(mype)

    for label, ax in zip('abcde', (ax1, ax2, ax3, ax4, ax5)):
        t = ax.text(0.02,0.95, f'{label})', transform=ax.transAxes, va='top',
                    fontsize=12)
        t.set_path_effects(mype)

    f.align_ylabels()
    plt.tight_layout()

    strOutFile = WORK_DIR / f"finetuning_curve_seed{nSeed}.jpg"
    plt.savefig(fname=strOutFile, dpi=300)
    plt.close()

# 
def drawNetForceAndAtomicDress(nSeed):

    import re
    from glob import glob
    import matplotlib.patheffects as pe

    pinnacle_output_dir = FT_DIR / f"output-cp2k-from-model-seed{nSeed}"

    # get net force results
    listNetForce = []
    listPiNNAppendFile = glob(f'{pinnacle_output_dir}/models/gen*/model1/append.log')
    listPiNNAppendGen = [int(re.search('gen(\d+)', l)[1]) for l in listPiNNAppendFile]
    for strFile in listPiNNAppendFile:
        with open(strFile, "r") as f:
            listLines = f.readlines()
            assert "Net force per atom (meV/Ang/Atom) on whole set" in listLines[0]
            dNetForcePerAtom_X = float(re.search(r"x=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", listLines[0])[1])
            dNetForcePerAtom_Y = float(re.search(r"y=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", listLines[0])[1])
            dNetForcePerAtom_Z = float(re.search(r"z=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", listLines[0])[1])
            listNetForce.append([dNetForcePerAtom_X, dNetForcePerAtom_Y, dNetForcePerAtom_Z])
    arrDataNetForce = np.array([[g,*e] for g, e in zip(listPiNNAppendGen, listNetForce)])
    arrDataNetForce = arrDataNetForce[arrDataNetForce[:,0].argsort()]

    # get e_dress changes over generations
    import yaml
    nMaxGen = np.max(listPiNNAppendGen)
    listPiNNYMLFile = glob(f'{pinnacle_output_dir}/models/gen{nMaxGen}/model1/model/params.yml*')
    listSavedTime = []
    listHAtomicDress = []
    listOAtomicDress = []
    for strFile in listPiNNYMLFile:

        listNameItems = strFile.split("/")[-1].split(".")
        if len(listNameItems) < 3:
            listSavedTime.append(np.nan)
        else:
            listSavedTime.append(int(listNameItems[-1]))

        params = {}
        with open(strFile, 'r') as f:
            params = yaml.load(f, Loader=yaml.Loader)

        dictDress = params["model"]["params"]["e_dress"]
        listHAtomicDress.append(dictDress[1])
        listOAtomicDress.append(dictDress[8])

    arrHAtomicDress = np.array([[g,e] for g, e in zip(listSavedTime, listHAtomicDress)])
    arrHAtomicDress = arrHAtomicDress[arrHAtomicDress[:,0].argsort()]

    arrOAtomicDress = np.array([[g,e] for g, e in zip(listSavedTime, listOAtomicDress)])
    arrOAtomicDress = arrOAtomicDress[arrOAtomicDress[:,0].argsort()]

    # do the plot
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[7, 5], 
                                    sharex=True, gridspec_kw={'hspace':0})

    l1, = ax1.plot(arrDataNetForce[:,0], arrDataNetForce[:,1], '.-')
    l2, = ax1.plot(arrDataNetForce[:,0], arrDataNetForce[:,2], 'd-')
    l3, = ax1.plot(arrDataNetForce[:,0], arrDataNetForce[:,3], 's-')
    l4, = ax2.plot(arrDataNetForce[:,0], arrHAtomicDress[:,1][0:len(arrDataNetForce[:,0])], '.-')
    l5, = ax3.plot(arrDataNetForce[:,0], arrOAtomicDress[:,1][0:len(arrDataNetForce[:,0])], '.-')

    # threshold
    ax1.plot([0,arrDataNetForce[:,0][-1]], [1.0,1.0], 'k--', lw=1)

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.set_ylabel('Net force\n [meV/$\AA$/atom]')
    ax2.set_ylabel('Atomic dress\n [eV]')
    ax3.set_ylabel('Atomic dress\n [eV]')
    ax3.set_xlabel('Generation')

    leg1 = ax1.legend([l1, l2, l3], ['X', 'Y', 'Z'], loc=1, frameon=True)
    leg2 = ax2.legend([l4], ['H'],  loc=1, frameon=True)
    leg3 = ax3.legend([l5], ['O'],  loc=1, frameon=True)

    ax1.set_ylim(-0.3,1.1)
    # ax2.set_ylim(20,99)
    # ax3.set_ylim(0,9)

    mype = [pe.withStroke(linewidth=4, foreground='w')]
    for leg in [leg1, leg2, leg3]:
        for t in leg.get_texts():
            t.set_path_effects(mype)

    for label, ax in zip('abc', (ax1, ax2, ax3)):
        t = ax.text(0.02,0.95, f'{label})', transform=ax.transAxes, va='top',
                    fontsize=12)
        t.set_path_effects(mype)

    f.align_ylabels()
    plt.tight_layout()

    strOutFile = WORK_DIR / f"net_force_e_dress_seed{nSeed}.jpg"
    plt.savefig(fname=strOutFile, dpi=300)
    plt.close()


if __name__ == '__main__':

    drawFinetuningCurve(nSeed=1)
    drawNetForceAndAtomicDress(nSeed=1)

    drawFinetuningCurve(nSeed=2)
    drawNetForceAndAtomicDress(nSeed=2)

    drawFinetuningCurve(nSeed=4)
    drawNetForceAndAtomicDress(nSeed=4)
