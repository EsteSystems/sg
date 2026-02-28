"""AST construction for the .sg contract format.

Parses token streams from the lexer into GeneContract, PathwayContract,
and TopologyContract AST nodes.
"""
from __future__ import annotations

from sg.parser.lexer import Token, TokenType, tokenize
from sg.parser.types import (
    GeneFamily, BlastRadius,
    FieldDef, TypeDef, VerifyStep, FeedsDef,
    GeneContract, PathwayContract, TopologyContract,
    PathwayStep as ASTPathwayStep, ForStep, ConditionalStep,
    Dependency, TopologyResource,
)


class ParseError(Exception):
    def __init__(self, message: str, token: Token | None = None):
        if token:
            super().__init__(f"L{token.line}: {message}")
        else:
            super().__init__(message)
        self.token = token


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, ttype: TokenType, value: str | None = None) -> Token:
        tok = self.advance()
        if tok.type != ttype:
            raise ParseError(
                f"expected {ttype.name} but got {tok.type.name} ({tok.value!r})", tok
            )
        if value is not None and tok.value != value:
            raise ParseError(
                f"expected {value!r} but got {tok.value!r}", tok
            )
        return tok

    def skip_newlines(self) -> None:
        while self.peek().type == TokenType.NEWLINE:
            self.advance()

    def at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def at_section_keyword(self) -> bool:
        """Check if current token is a section keyword."""
        tok = self.peek()
        if tok.type != TokenType.KEYWORD:
            return False
        return tok.value in (
            "is", "risk", "does", "takes", "gives", "types",
            "before", "after", "fails when", "unhealthy when",
            "verify", "feeds", "steps", "requires", "on failure", "has",
        )

    # --- Top level ---

    def parse(self) -> GeneContract | PathwayContract | TopologyContract:
        self.skip_newlines()
        tok = self.peek()
        if tok.type != TokenType.KEYWORD:
            raise ParseError(f"expected 'gene', 'pathway', or 'topology'", tok)
        if tok.value == "gene":
            return self.parse_gene()
        elif tok.value == "pathway":
            return self.parse_pathway()
        elif tok.value == "topology":
            return self.parse_topology()
        else:
            raise ParseError(f"expected 'gene', 'pathway', or 'topology', got {tok.value!r}", tok)

    # --- Gene contract ---

    def parse_gene(self) -> GeneContract:
        self.expect(TokenType.KEYWORD, "gene")
        name = self.expect(TokenType.IDENTIFIER).value
        self.skip_newlines()

        family: GeneFamily | None = None
        risk: BlastRadius = BlastRadius.NONE
        does = ""
        takes: list[FieldDef] = []
        gives: list[FieldDef] = []
        types: list[TypeDef] = []
        before: list[str] = []
        after: list[str] = []
        fails_when: list[str] = []
        unhealthy_when: list[str] = []
        verify: list[VerifyStep] = []
        verify_within: str | None = None
        feeds: list[FeedsDef] = []

        # Expect an INDENT for the body
        if self.peek().type == TokenType.INDENT:
            self.advance()

        while not self.at_end() and self.peek().type != TokenType.DEDENT:
            self.skip_newlines()
            if self.at_end() or self.peek().type == TokenType.DEDENT:
                break

            tok = self.peek()
            if tok.type != TokenType.KEYWORD:
                # Skip non-keyword lines
                self._skip_to_newline()
                continue

            section = tok.value
            self.advance()  # consume the keyword

            if section == "is":
                family = self._parse_family()
            elif section == "risk":
                risk = self._parse_risk()
            elif section == "does":
                self.expect(TokenType.COLON)
                self._consume_newline()
                does = self._parse_prose_block()
            elif section == "takes":
                self.expect(TokenType.COLON)
                self._consume_newline()
                takes = self._parse_field_defs()
            elif section == "gives":
                self.expect(TokenType.COLON)
                self._consume_newline()
                gives = self._parse_field_defs()
            elif section == "types":
                self.expect(TokenType.COLON)
                self._consume_newline()
                types = self._parse_type_defs()
            elif section == "before":
                self.expect(TokenType.COLON)
                self._consume_newline()
                before = self._parse_bullet_list()
            elif section == "after":
                self.expect(TokenType.COLON)
                self._consume_newline()
                after = self._parse_bullet_list()
            elif section == "fails when":
                self.expect(TokenType.COLON)
                self._consume_newline()
                fails_when = self._parse_bullet_list()
            elif section == "unhealthy when":
                self.expect(TokenType.COLON)
                self._consume_newline()
                unhealthy_when = self._parse_bullet_list()
            elif section == "verify":
                self.expect(TokenType.COLON)
                self._consume_newline()
                verify, verify_within = self._parse_verify_block()
            elif section == "feeds":
                self.expect(TokenType.COLON)
                self._consume_newline()
                feeds = self._parse_feeds_block()
            else:
                self._skip_to_newline()

        if self.peek().type == TokenType.DEDENT:
            self.advance()

        if family is None:
            raise ParseError("gene contract missing 'is' declaration")

        return GeneContract(
            name=name,
            family=family,
            risk=risk,
            does=does,
            takes=takes,
            gives=gives,
            types=types,
            before=before,
            after=after,
            fails_when=fails_when,
            unhealthy_when=unhealthy_when,
            verify=verify,
            verify_within=verify_within,
            feeds=feeds,
        )

    # --- Pathway contract ---

    def parse_pathway(self) -> PathwayContract:
        self.expect(TokenType.KEYWORD, "pathway")
        name = self.expect(TokenType.IDENTIFIER).value
        self.skip_newlines()

        risk: BlastRadius = BlastRadius.NONE
        does = ""
        takes: list[FieldDef] = []
        steps: list[ASTPathwayStep | ForStep | ConditionalStep] = []
        requires: list[Dependency] = []
        verify: list[VerifyStep] = []
        verify_within: str | None = None
        on_failure = "rollback all"

        if self.peek().type == TokenType.INDENT:
            self.advance()

        while not self.at_end() and self.peek().type != TokenType.DEDENT:
            self.skip_newlines()
            if self.at_end() or self.peek().type == TokenType.DEDENT:
                break

            tok = self.peek()
            if tok.type != TokenType.KEYWORD:
                self._skip_to_newline()
                continue

            section = tok.value
            self.advance()

            if section == "risk":
                risk = self._parse_risk()
            elif section == "does":
                self.expect(TokenType.COLON)
                self._consume_newline()
                does = self._parse_prose_block()
            elif section == "takes":
                self.expect(TokenType.COLON)
                self._consume_newline()
                takes = self._parse_field_defs()
            elif section == "steps":
                self.expect(TokenType.COLON)
                self._consume_newline()
                steps = self._parse_steps()
            elif section == "requires":
                self.expect(TokenType.COLON)
                self._consume_newline()
                requires = self._parse_requires()
            elif section == "verify":
                self.expect(TokenType.COLON)
                self._consume_newline()
                verify, verify_within = self._parse_verify_block()
            elif section == "on failure":
                self.expect(TokenType.COLON)
                self._consume_newline()
                on_failure = self._parse_on_failure()
            else:
                self._skip_to_newline()

        if self.peek().type == TokenType.DEDENT:
            self.advance()

        return PathwayContract(
            name=name,
            risk=risk,
            does=does,
            takes=takes,
            steps=steps,
            requires=requires,
            verify=verify,
            verify_within=verify_within,
            on_failure=on_failure,
        )

    # --- Topology contract ---

    def parse_topology(self) -> TopologyContract:
        self.expect(TokenType.KEYWORD, "topology")
        name = self.expect(TokenType.IDENTIFIER).value
        self.skip_newlines()

        does = ""
        takes: list[FieldDef] = []
        has: list[TopologyResource] = []
        verify: list[VerifyStep] = []
        verify_within: str | None = None
        on_failure = "preserve what works"

        if self.peek().type == TokenType.INDENT:
            self.advance()

        while not self.at_end() and self.peek().type != TokenType.DEDENT:
            self.skip_newlines()
            if self.at_end() or self.peek().type == TokenType.DEDENT:
                break

            tok = self.peek()
            if tok.type != TokenType.KEYWORD:
                self._skip_to_newline()
                continue

            section = tok.value
            self.advance()

            if section == "does":
                self.expect(TokenType.COLON)
                self._consume_newline()
                does = self._parse_prose_block()
            elif section == "takes":
                self.expect(TokenType.COLON)
                self._consume_newline()
                takes = self._parse_field_defs()
            elif section == "has":
                self.expect(TokenType.COLON)
                self._consume_newline()
                has = self._parse_has_block()
            elif section == "verify":
                self.expect(TokenType.COLON)
                self._consume_newline()
                verify, verify_within = self._parse_verify_block()
            else:
                self._skip_to_newline()

        if self.peek().type == TokenType.DEDENT:
            self.advance()

        return TopologyContract(
            name=name,
            does=does,
            takes=takes,
            has=has,
            verify=verify,
            verify_within=verify_within,
            on_failure=on_failure,
        )

    # --- Section parsers ---

    def _parse_family(self) -> GeneFamily:
        tok = self.advance()
        try:
            return GeneFamily(tok.value)
        except ValueError:
            raise ParseError(f"unknown gene family: {tok.value!r}", tok)

    def _parse_risk(self) -> BlastRadius:
        tok = self.advance()
        try:
            return BlastRadius(tok.value)
        except ValueError:
            raise ParseError(f"unknown risk level: {tok.value!r}", tok)

    def _parse_prose_block(self) -> str:
        """Parse a prose block (lines after 'does:'). Collected as STRING tokens by the lexer."""
        lines: list[str] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                tok = self.peek()
                if tok.type == TokenType.STRING:
                    lines.append(tok.value)
                    self.advance()
                elif tok.type == TokenType.NEWLINE:
                    self.advance()
                else:
                    break

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return "\n".join(lines).strip()

    def _parse_field_defs(self) -> list[FieldDef]:
        """Parse field definitions in takes/gives blocks.

        Format: name  type  "description"
        Optional: type? (optional), type = default
        """
        fields: list[FieldDef] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                tok = self.peek()
                if tok.type != TokenType.IDENTIFIER:
                    self._skip_to_newline()
                    continue

                name = self.advance().value
                field_type = self._read_type()

                optional = False
                if self.peek().type == TokenType.QUESTION:
                    self.advance()
                    optional = True

                default = None
                if self.peek().type == TokenType.EQUALS:
                    self.advance()
                    default = self._read_value()
                    optional = True

                description = ""
                if self.peek().type == TokenType.STRING:
                    description = self.advance().value

                fields.append(FieldDef(
                    name=name,
                    type=field_type,
                    required=not optional,
                    default=default,
                    optional=optional,
                    description=description,
                ))

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return fields

    def _parse_type_defs(self) -> list[TypeDef]:
        """Parse inline type definitions."""
        type_defs: list[TypeDef] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                tok = self.peek()
                if tok.type != TokenType.IDENTIFIER:
                    self._skip_to_newline()
                    continue

                type_name = self.advance().value
                if self.peek().type == TokenType.COLON:
                    self.advance()
                self._consume_newline()

                fields = self._parse_field_defs()
                type_defs.append(TypeDef(name=type_name, fields=fields))

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return type_defs

    def _parse_bullet_list(self) -> list[str]:
        """Parse a list of '- text' items."""
        items: list[str] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                if self.peek().type == TokenType.DASH:
                    self.advance()
                    text = self._read_rest_of_line()
                    items.append(text)
                else:
                    self._skip_to_newline()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return items

    def _parse_verify_block(self) -> tuple[list[VerifyStep], str | None]:
        """Parse verify block: locus param=value lines + optional 'within Ns'."""
        steps: list[VerifyStep] = []
        within: str | None = None

        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                tok = self.peek()

                # "within 30s"
                if tok.type == TokenType.KEYWORD and tok.value == "within":
                    self.advance()
                    within = self._read_duration()
                    continue

                if tok.type == TokenType.IDENTIFIER:
                    locus = self.advance().value
                    params: dict[str, str] = {}
                    while self.peek().type not in (
                        TokenType.NEWLINE, TokenType.EOF,
                        TokenType.DEDENT,
                    ):
                        if self.peek().type == TokenType.IDENTIFIER:
                            pname = self.advance().value
                            if self.peek().type == TokenType.EQUALS:
                                self.advance()
                                pval = self._read_param_value()
                                params[pname] = pval
                            else:
                                params[pname] = "true"
                        else:
                            self.advance()
                    steps.append(VerifyStep(locus=locus, params=params))
                else:
                    self._skip_to_newline()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return steps, within

    def _parse_feeds_block(self) -> list[FeedsDef]:
        """Parse feeds block: target_locus  timescale lines."""
        feeds: list[FeedsDef] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                if self.peek().type == TokenType.IDENTIFIER:
                    target = self.advance().value
                    timescale = "convergence"
                    if self.peek().type == TokenType.IDENTIFIER:
                        timescale = self.advance().value
                    feeds.append(FeedsDef(target_locus=target, timescale=timescale))
                else:
                    self._skip_to_newline()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return feeds

    def _parse_steps(self) -> list[ASTPathwayStep | ForStep | ConditionalStep]:
        """Parse pathway steps block."""
        steps: list[ASTPathwayStep | ForStep | ConditionalStep] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                tok = self.peek()

                # Step number: "1." or "2."
                if tok.type == TokenType.NUMBER:
                    index = int(self.advance().value)
                    if self.peek().type == TokenType.DOT:
                        self.advance()

                    self.skip_newlines()

                    # Check for "for" loop
                    if self.peek().type == TokenType.KEYWORD and self.peek().value == "for":
                        step = self._parse_for_step(index)
                        steps.append(step)
                        continue

                    # Check for "when" conditional
                    if self.peek().type == TokenType.KEYWORD and self.peek().value == "when":
                        step = self._parse_when_step(index)
                        steps.append(step)
                        continue

                    # Check for "->" pathway reference
                    is_ref = False
                    if self.peek().type == TokenType.ARROW:
                        self.advance()
                        is_ref = True

                    if self.peek().type == TokenType.IDENTIFIER:
                        locus = self.advance().value
                    else:
                        self._skip_to_newline()
                        continue

                    # Parse parameter bindings on subsequent indented lines
                    params = self._parse_step_params()

                    steps.append(ASTPathwayStep(
                        index=index,
                        locus=locus,
                        is_pathway_ref=is_ref,
                        params=params,
                    ))

                elif tok.type == TokenType.KEYWORD and tok.value == "for":
                    # for without a step number
                    step = self._parse_for_step(len(steps) + 1)
                    steps.append(step)
                else:
                    self._skip_to_newline()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return steps

    def _parse_for_step(self, index: int) -> ForStep:
        """Parse: for variable in {iterable}: -> locus ..."""
        self.expect(TokenType.KEYWORD, "for")
        variable = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.KEYWORD, "in")

        iterable = ""
        if self.peek().type == TokenType.REFERENCE:
            iterable = self.advance().value
        elif self.peek().type == TokenType.IDENTIFIER:
            iterable = self.advance().value

        if self.peek().type == TokenType.COLON:
            self.advance()

        self._consume_newline()

        # Parse body (indented lines after the for)
        body: ASTPathwayStep | None = None
        if self.peek().type == TokenType.INDENT:
            self.advance()
            self.skip_newlines()

            is_ref = False
            if self.peek().type == TokenType.ARROW:
                self.advance()
                is_ref = True

            if self.peek().type == TokenType.IDENTIFIER:
                locus = self.advance().value
                params = self._parse_step_params()
                body = ASTPathwayStep(
                    index=0,
                    locus=locus,
                    is_pathway_ref=is_ref,
                    params=params,
                )

            # Consume remaining body
            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self._skip_to_newline()
                self.skip_newlines()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return ForStep(
            index=index,
            variable=variable,
            iterable=iterable,
            body=body,
        )

    def _parse_when_step(self, index: int) -> ConditionalStep:
        """Parse: when step N.field: "value" -> locus ..."""
        self.expect(TokenType.KEYWORD, "when")
        self.expect(TokenType.KEYWORD, "step")
        step_num = int(self.expect(TokenType.NUMBER).value)
        self.expect(TokenType.DOT)
        field_name = self.expect(TokenType.IDENTIFIER).value

        if self.peek().type == TokenType.COLON:
            self.advance()

        self._consume_newline()

        branches: dict[str, ASTPathwayStep] = {}
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                # Parse branch: "value" -> locus or "value" locus
                branch_value = ""
                if self.peek().type == TokenType.STRING:
                    branch_value = self.advance().value
                elif self.peek().type == TokenType.IDENTIFIER:
                    branch_value = self.advance().value
                else:
                    self._skip_to_newline()
                    continue

                # Arrow is a visual connector in when branches, not a pathway ref
                if self.peek().type == TokenType.ARROW:
                    self.advance()

                if self.peek().type == TokenType.IDENTIFIER:
                    locus = self.advance().value
                    params = self._parse_step_params()
                    branches[branch_value] = ASTPathwayStep(
                        index=0,
                        locus=locus,
                        is_pathway_ref=False,
                        params=params,
                    )
                else:
                    self._skip_to_newline()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return ConditionalStep(
            index=index,
            condition_step=step_num,
            condition_field=field_name,
            branches=branches,
        )

    def _parse_step_params(self) -> dict[str, str]:
        """Parse parameter bindings on indented lines after a step locus."""
        params: dict[str, str] = {}
        self._consume_newline()

        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                if self.peek().type == TokenType.IDENTIFIER:
                    pname = self.advance().value
                    if self.peek().type == TokenType.EQUALS:
                        self.advance()
                        pval = self._read_param_value()
                        params[pname] = pval
                    self._consume_newline()
                else:
                    self._skip_to_newline()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return params

    def _parse_requires(self) -> list[Dependency]:
        """Parse requires block: 'step N needs step M' lines."""
        deps: list[Dependency] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                if self.peek().type == TokenType.KEYWORD and self.peek().value == "step":
                    self.advance()
                    step_num = int(self.expect(TokenType.NUMBER).value)
                    self.expect(TokenType.KEYWORD, "needs")
                    self.expect(TokenType.KEYWORD, "step")
                    needs_num = int(self.expect(TokenType.NUMBER).value)
                    deps.append(Dependency(step=step_num, needs=needs_num))
                self._skip_to_newline()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return deps

    def _parse_on_failure(self) -> str:
        """Parse on failure block — collect all text."""
        lines: list[str] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()
            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                if self.peek().type == TokenType.NEWLINE:
                    self.advance()
                    continue
                lines.append(self._read_rest_of_line())
            if self.peek().type == TokenType.DEDENT:
                self.advance()
        return "\n".join(lines).strip() if lines else "rollback all"

    def _parse_has_block(self) -> list[TopologyResource]:
        """Parse topology 'has' block."""
        resources: list[TopologyResource] = []
        if self.peek().type == TokenType.INDENT:
            self.advance()

            while not self.at_end() and self.peek().type != TokenType.DEDENT:
                self.skip_newlines()
                if self.at_end() or self.peek().type == TokenType.DEDENT:
                    break

                if self.peek().type == TokenType.IDENTIFIER:
                    name = self.advance().value
                    if self.peek().type == TokenType.COLON:
                        self.advance()
                    self._consume_newline()

                    resource_type = ""
                    properties: dict[str, str] = {}

                    if self.peek().type == TokenType.INDENT:
                        self.advance()
                        while not self.at_end() and self.peek().type != TokenType.DEDENT:
                            self.skip_newlines()
                            if self.at_end() or self.peek().type == TokenType.DEDENT:
                                break
                            if (
                                self.peek().type == TokenType.KEYWORD
                                and self.peek().value == "is"
                            ):
                                self.advance()
                                if self.peek().type == TokenType.IDENTIFIER:
                                    resource_type = self.advance().value
                            elif self.peek().type == TokenType.IDENTIFIER:
                                key = self.advance().value
                                val = self._read_rest_of_line().strip()
                                properties[key] = val
                            else:
                                self._skip_to_newline()
                        if self.peek().type == TokenType.DEDENT:
                            self.advance()

                    resources.append(TopologyResource(
                        name=name,
                        resource_type=resource_type,
                        properties=properties,
                    ))
                else:
                    self._skip_to_newline()

            if self.peek().type == TokenType.DEDENT:
                self.advance()

        return resources

    # --- Helpers ---

    def _read_duration(self) -> str:
        """Read a duration like '30s', '5m', '1h'. NUMBER + IDENTIFIER concatenated."""
        parts: list[str] = []
        while self.peek().type in (TokenType.NUMBER, TokenType.IDENTIFIER):
            parts.append(self.advance().value)
        return "".join(parts)

    def _read_type(self) -> str:
        """Read a type expression: 'string', 'string[]', 'mac_flap[]', 'int'."""
        if self.peek().type == TokenType.IDENTIFIER:
            t = self.advance().value
            # Check for array suffix
            if (
                self.peek().type == TokenType.IDENTIFIER
                and self.peek().value == "[]"
            ):
                self.advance()
                t += "[]"
            return t
        elif self.peek().type == TokenType.KEYWORD:
            # Some types like "int" might clash with keywords
            return self.advance().value
        return "unknown"

    def _read_value(self) -> str:
        """Read a value after '=': identifier, string, number, bool."""
        tok = self.peek()
        if tok.type == TokenType.STRING:
            return self.advance().value
        elif tok.type == TokenType.NUMBER:
            return self.advance().value
        elif tok.type == TokenType.IDENTIFIER:
            return self.advance().value
        elif tok.type == TokenType.KEYWORD:
            return self.advance().value
        return ""

    def _read_param_value(self) -> str:
        """Read a parameter value: {reference}, string, identifier, or composite."""
        tok = self.peek()
        if tok.type == TokenType.REFERENCE:
            return "{" + self.advance().value + "}"
        elif tok.type == TokenType.STRING:
            return self.advance().value
        elif tok.type == TokenType.NUMBER:
            return self.advance().value
        elif tok.type == TokenType.IDENTIFIER:
            val = self.advance().value
            # Handle hyphenated values like "active-backup"
            while self.peek().type == TokenType.DASH:
                self.advance()
                if self.peek().type == TokenType.IDENTIFIER:
                    val += "-" + self.advance().value
            return val
        return ""

    def _read_rest_of_line(self) -> str:
        """Read all remaining tokens on the current line as text."""
        parts: list[str] = []
        while self.peek().type not in (TokenType.NEWLINE, TokenType.EOF, TokenType.DEDENT):
            tok = self.advance()
            if tok.type == TokenType.STRING:
                parts.append(f'"{tok.value}"')
            elif tok.type == TokenType.REFERENCE:
                parts.append(f"{{{tok.value}}}")
            elif tok.type == TokenType.ARROW:
                parts.append("->")
            else:
                parts.append(tok.value)
        return " ".join(parts)

    def _consume_newline(self) -> None:
        """Consume a newline if present."""
        if self.peek().type == TokenType.NEWLINE:
            self.advance()

    def _skip_to_newline(self) -> None:
        """Skip tokens until the next newline."""
        while self.peek().type not in (TokenType.NEWLINE, TokenType.EOF):
            self.advance()
        if self.peek().type == TokenType.NEWLINE:
            self.advance()


def parse_sg(source: str) -> GeneContract | PathwayContract | TopologyContract:
    """Parse a .sg source string into a contract AST node."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()
