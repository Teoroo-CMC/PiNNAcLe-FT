nextflow.enable.dsl=2

params.publish = 'tools'

process release_space {
  tag "$name"
  label 'tools'
  publishDir "$params.publish/$name"

  input:
    tuple (val(root_dir),val(gen),path(output_dir))

  script:
    """
python3 - <<- 'EOF'
import os
import shutil
import numpy as np

# safe root to avoid deleting other directory
SAFE_ROOT = "$root_dir"

strWorkDir = "$output_dir"
strWorkDir = os.path.abspath(strWorkDir)
if not os.path.realpath(strWorkDir).startswith(SAFE_ROOT):
    raise RuntimeError(f"Unsafe work dir: {strWorkDir}")

nKeepGenNum = 1 # at least 1
listProcDir = ["models", "md", "collect", "label", "check", "merge", "dsmix"]

# delete the link file and target file
def removeSymlinkAndTarget(strItemPath):

    if not os.path.islink(strItemPath):
        return
    
    # relative path
    strTargetPath = os.readlink(strItemPath)
    if not os.path.isabs(strTargetPath):
        strLinkDir = os.path.dirname(strItemPath)
        strTargetPath = os.path.join(strLinkDir, strTargetPath)

    strTargetPath = os.path.realpath(strTargetPath)
    if not os.path.exists(strTargetPath):
        return

    # remove the symbol link
    os.unlink(strItemPath)

    # remove the real path/file
    if os.path.isfile(strTargetPath):
        os.remove(strTargetPath)
    elif os.path.isdir(strTargetPath):
        shutil.rmtree(strTargetPath)

# there is no need to keep the intermediate pinn.log, since the final pinn.log file includes all the information
def recursiveRemove(strSubDir):
    if not os.path.isdir(strSubDir):
        return
    listItems = os.listdir(strSubDir)
    for strItem in listItems:
        if "pinn.log" in strItem or "append.log" in strItem or "asemd.log" in strItem: # keep pinn.log and asemd.log
            continue
        strItemPath = os.path.join(strSubDir, strItem)
        if os.path.islink(strItemPath):
            removeSymlinkAndTarget(strItemPath)
        elif os.path.isfile(strItemPath):
            os.remove(strItemPath)
        elif os.path.isdir(strItemPath):
            recursiveRemove(strItemPath)

#
for strProcDir in listProcDir:

    strProcessDir = os.path.join(strWorkDir, strProcDir)
    if not os.path.isdir(strProcessDir):
        continue
    listSubDir = os.listdir(strProcessDir)

    # get the latest generation number
    listGenNum = [int(item.strip("gen")) for item in listSubDir]
    nLatestGen = int(np.max(listGenNum))
    listDelGen = list(range(1, nLatestGen-nKeepGenNum)) # keep gen0
    listDelSubDir = ["%d"%(gen) for gen in listDelGen] # for dsmix
    listDelSubDir += ["gen%d"%(gen) for gen in listDelGen] # for others

    for strSubDir in listDelSubDir:
        strSubDir = os.path.join(strProcessDir, strSubDir)
        if not os.path.exists(strSubDir):
            continue
        recursiveRemove(strSubDir)

EOF
    """
}

