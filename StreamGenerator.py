from StreamIntermediates import IslIR


class ISLGenerator:

    @classmethod
    def generate_concrete_repr(cls, ir: IslIR, accessed_array_name: str = ''):
        it_dom = ir.iteration_domain.to_abstract_repr(ir.name)
        maps = [map.to_abstract_repr(ir.iteration_domain.vector, ir.name, accessed_array_name)
                for map in ir.access_maps]
        


    def check_access_map_aliasing(self):
        # TODO: Implement
        pass
