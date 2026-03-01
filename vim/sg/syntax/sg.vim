" Vim syntax file for Software Genome .sg contract files
if exists('b:current_syntax')
  finish
endif

" --- Top-level declarations + contract name ---
" Declaration line matches the whole line; contains name for sub-highlighting
syn match sgDeclarationLine '\v^(gene|pathway|topology)\s+[a-zA-Z_][a-zA-Z0-9_-]*' contains=sgDeclaration,sgName
syn match sgDeclaration '\v^(gene|pathway|topology)\ze\s' contained
syn match sgName '\v\s\zs[a-zA-Z_][a-zA-Z0-9_-]*' contained

" --- Section headers (colon-terminated) ---
syn match sgSection '\v^\s+\zs(does|takes|gives|types|before|after|verify|steps|requires|has)\ze\s*:'
syn match sgSection '\v^\s+\zsfails\s+when\ze\s*:'
syn match sgSection '\v^\s+\zsunhealthy\s+when\ze\s*:'
syn match sgSection '\v^\s+\zson\s+failure\ze\s*:'

" --- Modifier keywords ---
syn match sgModifier '\v^\s+\zsis\ze\s'
syn match sgModifier '\v^\s+\zsrisk\ze\s'
syn match sgModifier '\v^\s+\zswithin\ze\s'

" --- Constants (must come after modifiers for priority) ---
" Risk levels (after 'risk' keyword)
syn match sgConstant '\v<(none|low|medium|high|critical)>$'
" Gene families (after 'is' keyword)
syn match sgConstant '\v<(configuration|diagnostic)>$'

" --- Control flow ---
syn match sgControl '\v<for>\ze\s'
syn match sgControl '\v\s\zsin\ze\s'
syn match sgControl '\v<when>\ze\s'
syn match sgControl '\v<step>\ze\s'
syn match sgControl '\v\s\zsneeds\ze\s'

" --- Types ---
syn match sgType '\v<(string|bool|int|float)(\[\])?\??'

" --- Boolean values ---
syn keyword sgBoolean true false enabled disabled

" --- Strings ---
syn region sgString start='"' end='"' oneline

" --- References {param_name} ---
syn region sgReference start='{' end='}' oneline

" --- Operators ---
syn match sgArrow '->'
syn match sgOperator '\v\s\zs\=\ze\s'

" --- Step numbers (e.g. "    1. gene_name") ---
syn match sgStepNumber '\v^\s+\zs\d+\.\ze\s'

" --- Duration literals (e.g. "30s", "5m", "1h") ---
syn match sgDuration '\v\d+[smh]\s*$'

" --- Bullet list markers ---
syn match sgBullet '\v^\s+\zs-\ze\s'

" --- Section colons ---
syn match sgDelimiter '\v(does|takes|gives|types|before|after|verify|steps|requires|has|when|failure)\s*\zs:'

" --- Link to standard highlight groups ---
hi def link sgDeclaration Keyword
hi def link sgName Function
hi def link sgSection Statement
hi def link sgModifier Keyword
hi def link sgControl Conditional
hi def link sgType Type
hi def link sgConstant Constant
hi def link sgBoolean Boolean
hi def link sgString String
hi def link sgReference Special
hi def link sgArrow Operator
hi def link sgOperator Operator
hi def link sgStepNumber Number
hi def link sgDuration Number
hi def link sgBullet Delimiter
hi def link sgDelimiter Delimiter

let b:current_syntax = 'sg'
