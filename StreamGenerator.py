from StreamIntermediates import IslIR
from typing import Tuple

class ISLGenerator:

    @classmethod
    def generate_concrete_repr(cls, arg_vals: Tuple, ir: IslIR, accessed_array_name: str = ''):
        ir.invariants.eval(ir.arguments, arg_vals)
        for map in ir.access_maps:
            map_structure = map.to_abstract_repr(
                ir.iteration_domain, ir.name, accessed_array_name)

            # colocate access expr paramameters to either stream arguments or invariants
            # colocate access conditional parameters to either stream arguments or invariants
            # Replace map access expr parameters with .format
            # conver map conditional parameters to set with {:} context
            # apply map conditional parameters with set intersect
            # if any parameters are actually invariants evaluate invariants with compile and eval as a function with a return

        it_dom = ir.iteration_domain.to_abstract_repr(ir.name)
        # colocate it_domain parameters with either stream arguments or invariants
        # same as above, if you see invariants evaulate them with compile and eval
        
    def check_access_map_aliasing(self):
        # TODO: Implement
        pass
