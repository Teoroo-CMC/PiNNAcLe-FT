from pathlib import Path

########## this is a default setting file for the project #############

PROJ_DIR = Path("C:\\TeC-ZhanYunZhang\\0Software\\1MyPythonCode\\PiNNAcLe-Proton-aq\\")

proj_params = {
    "proj_dir"            : PROJ_DIR,
    "dataset_dir"         : PROJ_DIR / "1_Datasets",
    "pinet2_init_dir"     : PROJ_DIR / "2_PiNet2_init",
    "pinet2_ft_dir"       : PROJ_DIR / "3_PiNet2_ft",
    "equ_md_dir"          : PROJ_DIR / "4_MD_equ",
    "nonequ_md_dir"       : PROJ_DIR / "5_MD_nonequ",
    "plot_dir"            : PROJ_DIR / "6_Plotting",
}

#
plot_params = {
    "default_font_size"   : 10.0,
    "resolution_dpi"      : 300.0,
    "single_col_width"    : 3.30, # inch, 88 mm, 3.46457 for nat comm, 3.3 for ACS
    "double_col_width"    : 7.00, # inch, 180 mm, 7.08661 for nat comm. 7.0 for ACS
    "figure_dir"          : proj_params["plot_dir"],
}
