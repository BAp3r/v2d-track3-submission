"""Arguments shared by both ego entrypoints."""
def add_ace_args(parser):
    parser.add_argument("--hand_detector", choices=("wilor", "ace"), default="wilor",
                        help="Replace the WiLoR detection/MANO component with ACE")
    parser.add_argument("--ace_weights", default="data/weights/ace_hand")
    parser.add_argument("--ace_backbone", default="data/weights/wan_control")
    parser.add_argument("--ace_mano", default="data/weights/hand/mano")
    parser.add_argument("--ace_cache", default="data/ace_cache")
    parser.add_argument("--ace_native_predictions", default=None,
                        help="Optional official native prediction pickle for verified CPU re-export")
    parser.add_argument("--ace_input_intrinsics", default=None,
                        help="Optional calibrated camera JSON for ACE; defaults to pipeline intrinsics")
    parser.add_argument("--skip_hand_overlays", action="store_true",
                        help="Skip WiLoR/ACE diagnostic hand mesh/mask MP4s; keep all predictions and result export")
