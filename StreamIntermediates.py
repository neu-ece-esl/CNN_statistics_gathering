
class ISLAbstractRepresentation:
    # TODO: Implement
    def __init__(self, ir: IslIR) -> None:
        #    self.iteration_domain = isl.BasicSet(iteration_domain.)
        pass


@dataclass
class ISLConcreteRepresentation:
    # TODO: Implement
    pass


@dataclass
class IslIR:
    name: str = ""
    iteration_domain: IterationDomain = IterationDomain()
    access_maps: Tuple[AccessMap] = ()
    invariants: Tuple[Invariant] = ()
    arguments: Dict[str, Union[int, None]] = field(default_factory=dict)




@dataclass
class StreamTokens:
    name: str = ""
    start_time: int = 0
    accessed_array: str = ""
    generator_args: ast.arguments = None
    yield_exprs: Tuple[Union[ast.Yield, Tuple[Union[ast.If, ast.Yield]]]] = ()
    invariant_assignments: Tuple[ast.Assign] = ()
    for_loops: Tuple[ast.For] = ()