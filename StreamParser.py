
from StreamIntermediates import StreamTokens, IslIR
from StreamParserPrimitives import AccessMap
from StreamHelpers import NamedEntityAnnotator, NamedEntityExtractor
import ast
import astor
import re
class StreamParser:

    symbol_conversion_table = {r"==": "=", r"\n": "", r" +": " "}

    @classmethod
    def parse(cls, tokens: StreamTokens):
        parser = cls()
        parser.parse_name(tokens)
        parser.parse_arguments(tokens)
        parser.parse_invariants(tokens)
        parser.parse_iteration_domain(tokens)
        parser.parse_access_maps(tokens)
        return parser.ir

    def __init__(self):
        self.ir = IslIR()

    def convert_non_isl_symbols_to_isl_equivelent(self, expr_str):
        for symbol, target in StreamParser.symbol_conversion_table.items():
            expr_str = re.sub(symbol, target, expr_str)
        return expr_str

    def wrap_with_expr_and_fix_ast_entity(self, entity):
        wrapped_entity = ast.fix_missing_locations(ast.Expression(entity))
        return wrapped_entity

    def convert_expr_to_str(self, expr):
        expr_str = astor.to_source(expr)
        expr_str = self.convert_non_isl_symbols_to_isl_equivelent(expr_str)
        return expr_str

    def convert_access_map_condition_to_expr(self, condition, annotate_params=False):
        condition_expr = self.wrap_with_expr_and_fix_ast_entity(condition)
        if annotate_params:
            condition_expr = NamedEntityAnnotator.annotate(
                condition_expr, ignore=self.ir.iteration_domain.vector
            )
        return condition_expr

    def parse_name(self, tokens: StreamTokens):
        self.ir.name = tokens.name.upper()

    def parse_arguments(self, tokens: StreamTokens):
        if tokens.generator_args.vararg is not None:
            raise SyntaxError(f"Varargs not allowed in stream template {self.ir.name}")
        if tokens.generator_args.kwarg is not None:
            raise SyntaxError(f"Kwargs not allowed in stream template {self.ir.name}")

        num_of_args_with_default_val = len(tokens.generator_args.defaults)
        if num_of_args_with_default_val != 0:
            arg_list = tokens.generator_args.args[:(-num_of_args_with_default_val)]
        else:
            arg_list = tokens.generator_args.args

        for arg in arg_list:
            self.ir.arguments[arg.arg] = None
        for arg, default in zip(
            reversed(arg_list), reversed(tokens.generator_args.defaults)
        ):
            if not isinstance(default, ast.Constant) or not isinstance(
                default.value, int
            ):
                raise SyntaxError(
                    f"Stream template arg: {arg.arg} has a non-int default: {default.value}"
                )
            self.ir.arguments[arg.arg] = default

    def parse_invariants(self, tokens: StreamTokens):
        invariant_targets = [
            assignment.targets[0].id for assignment in tokens.invariant_assignments
        ]
        for idx_current_assignment, assignment in enumerate(
            tokens.invariant_assignments
        ):
            target = assignment.targets[0].id
            if len(assignment.targets) > 1:
                raise SyntaxError(
                    f"Invariant assignment with target '{target}' can only have one target"
                )
            assignment_vars = NamedEntityExtractor.extract(assignment.value)
            for var in assignment_vars:
                if var not in invariant_targets and var not in self.ir.arguments.keys():
                    raise SyntaxError(
                        f"Invalid variable in invariant assignment target '{target}', '{var}' is not a stream template arg nor is it another invariant"
                    )
                if var in invariant_targets:
                    idx_of_var_in_targets = invariant_targets.index(var)
                    if idx_of_var_in_targets >= idx_current_assignment:
                        raise SyntaxError(
                            f"Invariant assignment '{target}' references '{var}' before its assignment"
                        )
        self.ir.invariants = tuple(invariant_targets)

    def parse_access_maps(self, tokens: StreamTokens):
        for expr in tokens.yield_exprs:
            condition_list = expr[:-1]
            yield_expr = expr[-1]

            # Get parameter list and validate
            chain_parameter_set = set()
            for condition in condition_list:
                for param in NamedEntityExtractor.extract(
                    condition.test, ignore=self.ir.iteration_domain.vector
                ):
                    if (
                        param not in self.ir.arguments
                        and param not in self.ir.invariants
                    ):
                        raise SyntaxError(
                            f"Parameter {param} in expression \n'{astor.to_source(condition.test).strip()}'\nis not a stream argument nor an invariant"
                        )
                    chain_parameter_set.add(param)

            for param in NamedEntityExtractor.extract(
                yield_expr, ignore=self.ir.iteration_domain.vector
            ):
                if param not in self.ir.arguments and param not in self.ir.invariants:
                    raise SyntaxError(
                        f"Parameter {param} in expression \n'{astor.to_source(yield_expr.value).strip()}'\nis not a stream argument nor an invariant"
                    )
                chain_parameter_set.add(param)

            # Get condition
            chain_conditions_expr = self.convert_expr_to_str(
                ast.BoolOp(
                    op=ast.And(),
                    values=[
                        self.convert_access_map_condition_to_expr(condition.test)
                        for condition in condition_list
                    ],
                )
            )

            # Get annotated condition
            chain_conditions_expr_with_annotated_params = self.convert_expr_to_str(
                ast.BoolOp(
                    op=ast.And(),
                    values=[
                        self.convert_access_map_condition_to_expr(
                            condition.test, annotate_params=True
                        )
                        for condition in condition_list
                    ],
                )
            )

            # Get access expression
            yield_expr_with_annotated_params = self.convert_expr_to_str(
                NamedEntityAnnotator.annotate(
                    yield_expr.value, ignore=self.ir.iteration_domain.vector
                )
            )
            yield_expr = self.convert_expr_to_str(yield_expr.value)

            self.ir.access_maps += (
                AccessMap(
                    access_expr=yield_expr,
                    access_expr_with_annotated_parameters=yield_expr_with_annotated_params,
                    condition=chain_conditions_expr,
                    condition_with_annotated_parameters=chain_conditions_expr_with_annotated_params,
                    parameters=chain_parameter_set,
                ),
            )

    def parse_iteration_domain(self, tokens: StreamTokens):
        # Get iteration domain vectors
        self.ir.iteration_domain.vector = tuple(
            [loop.target.id for loop in tokens.for_loops]
        )

        for loop in tokens.for_loops:
            if isinstance(loop.iter, ast.Call) and loop.iter.func.id == "range":
                for arg in loop.iter.args:
                    if isinstance(arg, ast.Name):
                        if (
                            arg.id not in self.ir.arguments
                            and arg not in self.ir.invariants
                        ):
                            raise SyntaxError(
                                f"Argument '{arg.id}' in loop \n'{astor.to_source(loop).strip()}'\nis not a stream argument nor an invariant"
                            )
                        self.ir.iteration_domain.parameters.add(arg.id)
                    # TODO: Handle non name instance e.g. function
                bounds = ()
                for arg in loop.iter.args[:2]:
                    try:
                        bounds += (arg.value,)
                    except AttributeError:
                        bounds += (arg.id,)
                    except AttributeError:
                        raise SyntaxError(
                            f"Invalid bound argument '{arg}' for loop range"
                        )
                self.ir.iteration_domain.bounds += (bounds,)
                if len(loop.iter.args) == 3:
                    arg = loop.iter.args[2]
                    try:
                        self.ir.iteration_domain.steps += (arg.value,)
                    except AttributeError:
                        raise SyntaxError(
                            f"Invalid step argument '{arg.id}' in for loop range, only constants are allowed"
                        )
            else:
                raise SyntaxError(
                    f"For loops with iterator '{loop.target.id}' can only iterate over ranges"
                )
