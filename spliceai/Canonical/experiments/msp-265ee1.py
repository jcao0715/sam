from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.architecture["splicepoint_model_spec"] = dict(
    type="BothLSSIModels",
    acceptor="model/splicepoint-model-acceptor-1",
    donor="model/splicepoint-model-donor-1",
)

msp.architecture["motif_model_spec"]["motif_width"] = 13
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="ResidualStack",
    depth=3,
)

msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["motif_model_spec"] = dict(
    type="AdjustedMotifModel",
    model_to_adjust_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbns"),
        exclude_names=("3P", "5P"),
    ),
    adjustment_model_spec=msp.architecture["motif_model_spec"],
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
)
msp.architecture["motif_model_spec"] = dict(
    type="ReprocessedMotifModel",
    motif_model_1_spec=msp.architecture["motif_model_spec"],
    reprocessor_spec=dict(
        type="ResidualStackSRMP",
        radius=12,
        channels=200,
        depth=3,
        influence_propagation_spec=dict(type="PropagateSparsityAndSum"),
    ),
)
msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.run()