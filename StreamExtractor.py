

class StreamExtractor(astor.ExplicitNodeVisitor):
    @classmethod
    def extract(cls, node, start_time=None, accessed_array=None):
        extractor = cls()
        extractor.visit(node)
        extractor.prune_orelse()
        return extractor.tokens

    def __init__(self):
        self.tokens = StreamTokens()

    def prune_orelse(self):
        for yield_expr in self.tokens.yield_exprs:
            for condition in yield_expr:
                if isinstance(condition, ast.If):
                    condition.orelse = []
                elif isinstance(condition, ast.Yield):
                    break
                else:
                    raise Exception(
                        "Found non condition or yield node when pruning yield expressions orelse conditions"
                    )

    def visit_Module(self, node: ast.Module) -> Any:
        stream_func_def_node = node.body[0]
        self.visit_FunctionDef(stream_func_def_node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.tokens.name = node.name
        if node.decorator_list[0].id != "stream":
            raise SyntaxError(
                f"Invalid function '{node.name}' used for conversion to ISL IR"
            )
        self.tokens.generator_args = node.args
        start_of_for_loops_idx = 0
        for idx, _node in enumerate(node.body):
            if isinstance(_node, ast.Assign):
                self.tokens.invariant_assignments += (_node,)
            elif isinstance(_node, ast.For):
                start_of_for_loops_idx = idx
                break
            else:
                raise SyntaxError(
                    f"Invalid expression '{astor.to_source(_node).strip()}' in '{node.name}' body. Only constant assignments, and for loops are allowed"
                )

        if start_of_for_loops_idx != len(node.body) - 1:
            logging.warning(
                "Other expressions in function definition beyond the first for loop will be ignored"
            )
        self.visit_For(node.body[start_of_for_loops_idx])

    def visit_For(self, node: ast.For) -> Any:
        self.tokens.for_loops += (node,)
        if len(node.body) > 1:
            logging.warning(
                f"Multiple statements in body of for loop with iterator '{node.target.id}', only the first statement will be parsed"
            )
        first_entry_in_loop_body = node.body[0]
        if isinstance(first_entry_in_loop_body, ast.For):
            self.visit_For(first_entry_in_loop_body)
        elif isinstance(first_entry_in_loop_body, ast.If):
            self.visit_If(first_entry_in_loop_body)
        elif isinstance(first_entry_in_loop_body, ast.Expr):
            if isinstance(first_entry_in_loop_body.value, ast.Yield):
                self.visit_Yield(first_entry_in_loop_body.value)
            else:
                raise SyntaxError(
                    f"Invalid expression '{astor.to_source(first_entry_in_loop_body).strip()}' in for loop body, only yield expressions are allowed"
                )
        else:
            raise SyntaxError(
                f"Invalid statement '{astor.to_source(first_entry_in_loop_body).strip()}' in for loop with iterator  '{node.target.id}' body, only other for loops, if statements, and yield expressions are allowed"
            )

    def visit_Yield(self, node: ast.Yield, if_chain: Tuple = ()) -> Any:
        if_chain += (deepcopy(node),)
        self.tokens.yield_exprs += deepcopy((if_chain,))

    def visit_If(self, node: ast.If, if_chain=()) -> Any:
        if_chain += (deepcopy(node),)
        if len(node.body) > 1:
            raise SyntaxError(
                f"Invalid additional expression in if/elif/else body \n'{astor.to_source(node.body[1]).strip()}'\n\nonly one expression is allowed in if/elif/else body"
            )
        if len(node.orelse) > 1:
            raise SyntaxError(
                f"Invalid additional expression in if/elif/else body \n'{astor.to_source(node.orelse[1]).strip()}'\n\nonly one expression is allowed in if/elif/else body"
            )

        if len(node.body) == 1:
            entry = node.body[0]
            if isinstance(entry, ast.If):
                self.visit_If(entry, if_chain)
            elif isinstance(entry, ast.Expr):
                if isinstance(entry.value, ast.Yield):
                    self.visit_Yield(entry.value, if_chain)
                else:
                    raise SyntaxError(
                        f"Invalid statement \n'{astor.to_source(entry)}'\n in if condition body, only yield expression are allowed"
                    )
            else:
                raise SyntaxError(
                    f"Invalid statement \n'{astor.to_source(entry)}'\n , only other if statements, and yield expressions are allowed"
                )
        if len(node.orelse) == 1:
            entry = node.orelse[0]
            if_chain[-1].test = ast.UnaryOp(op=ast.Not(), operand=if_chain[-1].test)
            if isinstance(entry, ast.If):
                self.visit_If(entry, if_chain)
            elif isinstance(entry, ast.Expr):
                if isinstance(entry.value, ast.Yield):
                    self.visit_Yield(entry.value, if_chain)
                else:
                    raise SyntaxError(
                        f"Invalid statement \n'{astor.to_source(entry)}'\n in if condition body, only yield expression are allowed"
                    )
            else:
                raise SyntaxError(
                    f"Invalid statement \n'{astor.to_source(entry)}'\n in if condition body, only other if statements, and yield expressions are allowed"
                )
