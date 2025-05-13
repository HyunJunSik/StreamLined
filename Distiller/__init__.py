from .distiller import Distiller
from .KD import KD
from .DKD import DKD
from .CLKD import CLKD
from .MLKD import MLKD_align_3, MLKD_align_4, MLKD_align_5

distiller_dict = {
    "KD" : KD,
}
