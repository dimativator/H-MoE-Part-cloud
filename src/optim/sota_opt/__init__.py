from .ademamix import AdEMAMix
from .ademamix_sign import AdEMAMixSign
from .fp8_ademamix import FP8AdEMAMix
from .fp8_soap import FP8SOAP
# from .muon import Muon, DistributedMuon
# from .rmsprop import RMSpropW
from .adopt import ADOPT
from .lion import Lion
# from .ns_ademamix import NSAdEMAMix
from .soap.soap_harvard import SOAP
from .MARS.MARS.optimizers.mars import MARS
from .MARS.MARS_M.optimizers.mars_m import MARS_M
from .swan import SWAN
try:
    from .shampoo.shampoo import DistributedShampoo
except ModuleNotFoundError as error:
    if error.name != "distributed_shampoo":
        raise
    DistributedShampoo = None
