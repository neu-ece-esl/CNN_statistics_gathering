
@dataclass
class IterationDomain:
    _vector: Tuple[str] = ()
    bounds: Tuple[Tuple[Union[int, str]]] = ()
    steps: Tuple[Tuple[Union[int, str]]] = ()
    parameters: Set[str] = field(default_factory=set)

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vector: Tuple):
        if len(vector) != len(set(vector)):
            for iterator, count in Counter(vector).items():
                if count > 1:
                    raise SyntaxError(f"Duplicate iterator {iterator} in For loops")
        for val in vector:
            if val == "_":
                raise SyntaxError("Anonymous iterators not allowed in for loops")
        self._vector = vector


    @classmethod
    def generate_abstract_repr(cls, name, iteration_domain):
        params = ISLGeneratorPreprocessor.preprocess_iterable(iteration_domain.parameters)
        it_vector = ISLGeneratorPreprocessor.preprocess_iterable(iteration_domain.vector)

        bound_tuples = list(zip(iteration_domain.vector, iteration_domain.bounds))
        bound_tuples_count = len(bound_tuples)

        bound_expr = ""
        for idx, (iterator, bound) in enumerate(bound_tuples):
            # upper bound
            if len(bound) == 1:
                bound_expr += f" 0 <= {iterator} < {bound[0]} "
            # upper and lower bound
            elif len(bound) == 2:
                bound_expr += f" {bound[0]} <= {iterator} < {bound[1]} "
            else:
                raise SyntaxError("Invalid number of iterator bounds")

            if idx < bound_tuples_count - 1:
                bound_expr += "and"

        step_adjustment = ""
        for idx, step in enumerate(iteration_domain.steps):
            step_adjustment += "and"
            step_adjustment += f" {iteration_domain.vector[idx]} mod {step} = 0 "

        structure = f"[{params}] -> {{ {name.upper()}[{it_vector}] : {bound_expr} {step_adjustment}}}"

        return isl.BasicSet(structure)


@dataclass
class AccessMap:
    access_expr: str = ""
    access_expr_with_annotated_parameters: str = ""
    condition: str = ""
    condition_with_annotated_parameters: str = ""
    parameters: Set[str] = field(default_factory=set)

    @classmethod
    def generate_abstract_repr(cls, name, access_map):
        params = ISLGeneratorPreprocessor.preprocess_iterable(access_map.parameters)
        
        if len(access_map.access_expr) != len(access_map.condition):
            raise Exception("Number of conditions doesn't match number of access expressions")

        for map, expr in zip(access_map.access_expr, access_map.condition):
            pass 

@dataclass
class Invariant:
    name: str = ""