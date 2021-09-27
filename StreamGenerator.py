

class ISLGenerator:

    @classmethod
    def generate_abstract_repr(cls, ir: IslIR):
        it_dom = ir.iteration_domain
        it_dom = IterationDomain.generate_abstract_repr(ir.name, it_dom) 

    @classmethod
    def generate_concrete_repr(ir):
        # TODO: Implement
        pass

    def check_access_map_aliasing(self):
        # TODO: Implement
        pass