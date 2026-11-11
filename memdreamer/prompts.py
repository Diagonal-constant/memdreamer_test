SYSTEM_VISION = "You are a precise video-analysis assistant. Follow the requested output format exactly. Write in English."

"""Prompts reconstructed from the paper's appendix (Figs 7-11), generalised beyond the
contest/sports examples. Placeholders use <<NAME>> and are filled with str.replace."""

SEGMENT_PROMPT = """You are an Expert Visual Story Segmenter.

[YOUR CORE OBJECTIVE]
You are shown frames sampled from ONE window of a long video, packed into mosaic images. Every tile has its ABSOLUTE video timestamp burned into its top-left corner and listed in the text before each image. The window runs <<START>>-<<END>>. The window is <<DUR>> long.
Segment it into "Major Story Episodes". Each episode MUST be a complete, self-contained logical event, typically 3 to 8 minutes (shorter only if the content genuinely changes, e.g. a distinct news item or scene).

[STRICT SEGMENTATION RULES]
1. WHAT CONSTITUTES AN EPISODE: a full narrative arc for ONE subject or ONE specific phase (e.g. one person's whole appearance from entering to leaving; one news story; one scene at one location). Do not split it.
2. RELY ON PURE VISUAL CUES to mark a new episode:
   - a completely new main subject / person / scene appearing;
   - distinct broadcast graphics (new name tag, lower-third, title card);
   - a major, permanent location change.
3. ANTI-FRAGMENTATION (DO NOT CUT): do not cut when the camera merely switches angle (speaker -> reaction shot -> audience), nor for flashing lights or brief effects while the core subject stays the same.
4. BOUNDARY PRECISION (NO BLEEDING): an episode's end_time must NOT include the first appearance of the next episode's subject. Stop right BEFORE the new subject / scene / graphic appears.
5. Episodes must be contiguous, non-overlapping and start at the window start (<<START>>). If the last episode is still in progress at the end of the window, end it at the last frame.

[CAST]
Also list every distinct PERSON who appears in the window ONCE, even if they appear in several episodes (staff, customers, children, bystanders). Same clothing + role = same person; do not split one person into several.

[OUTPUT FORMAT]
Output ONLY one valid JSON object, no markdown fences: {"cast": [...], "episodes": [...]}
- cast item: {"pid": "P1", "description": "role + clothing + distinguishing look, e.g. 'cashier, purple shirt'", "first_seen": "HH:MM:SS", "last_seen": "HH:MM:SS", "with": "companions' pids or null"}
- episodes item:
- "start_time": "HH:MM:SS" (absolute video time)
- "end_time": "HH:MM:SS" (absolute video time)
- "episode_title": concise name
- "visual_evidence": the visual cue that justified the boundary, confirming no bleed-over
"""

SUBGRAPH_PROMPT = """You are a video knowledge-graph annotator. Given a segment of video (frames packed into mosaic images; each tile carries its ABSOLUTE video timestamp), output a single JSON object describing its subgraph G_i = (V_i, E_i) with node groups V^E (entities), V^M (micro-events) and on-screen text, plus the edges among them. Look carefully before answering.

# Task overview
For this segment produce an exhaustive but de-duplicated list of:
1. micro-events: every meaningful action or happening, time-stamped at second precision (ABSOLUTE video seconds, read from the tile labels).
2. entities: every PERSON, OBJECT, LOCATION or GROUP that participates in or appears alongside those events.
3. on-screen text: captions, titles, lower-thirds, signs, scoreboards (verbatim).
4. edges: directed, labelled relations covering subject/object roles, spatial-attribute ties, temporal order and explicit causality.
Also write a label and a summary of the whole segment. The summary must state WHAT happens, in what ORDER and WHY (cause -> effect chains, purposes of actions), naming people/objects by their distinguishing attributes.

# Node schema
- micro_events: {event_id:"M1", event_type, time_range:[start_sec,end_sec], subject, object, action, description}
  * subject/object specific enough to match one entity; action is the precise motion;
  * description is the retrieval payload: the full sequence, with cause/purpose when applicable.
  * STATE CHANGES are micro-events too: an object opening/closing (cash drawer, door, lid, bag), appearing/arriving or leaving/disappearing, being picked up or put down. Name the object plainly (e.g. "cash drawer").
  * TRANSFERS: for every hand-over write giver -> receiver and WHAT exactly (cash notes, coins, change, card, phone, receipt, goods), e.g. subject "customer in white kurta", action "hands cash notes to", object "cashier". Paying and receiving change are different events.
  * Record only what is visible in the frames. Do not pad with evenly spaced generic events ("continues to interact"); a long unchanged activity is ONE event with a long time_range.
  * Do NOT create micro-events for static on-screen graphics.
- entities: {entity_id:"E1", name|null, entity_type in {PERSON,OBJECT,LOCATION,GROUP}, attributes, description, time_range:[first_seen_sec,last_seen_sec], visual_grounding}
  * time_range: first and last frame in THIS segment where the entity is visible (absolute seconds).
  * name only if explicitly given by a caption/graphic/speech label; never invent.
  * attributes: PERSON{clothing,hair,role,...}; OBJECT{type,color,location}; LOCATION/GROUP{name,type}.
  * description fuses attributes + visual cues in one sentence.
  * visual_grounding = {primary_time_sec:int, distinctive_features:[2-4 cues], spatial_hint}.
- on_screen_text: {text_id:"T1", text (verbatim), time_range:[start_sec,end_sec], description (what it means in context)}
  If the same overlay persists, emit ONE node spanning its full time_range.

# Edge schema: {source_id, target_id, relation_label, description}. Allowed labels:
- Entity -> Event: PERFORMS, RECEIVES, USES_TOOL
- Event -> Entity: HAPPENS_IN
- Text -> Event: CONTEXT_FOR ; Text -> Entity: ANNOTATES
- Event -> Event: BEFORE, OVERLAP, CAUSES, PREVENTS, SUBEVENT_OF
- Entity <-> Entity: PART_OF, LOCATED_IN, NEXT_TO, ATTACHED_TO, IS_SAME_AS

# People in this video (cast from a whole-window pass)
<<CAST>>
For a PERSON entity that matches a cast member, set "name" to that pid + description exactly as listed (e.g. "P2 man in white kurta"), and use the same text in subject/object. Only create a new person if nobody in the cast matches.

# Continuity with earlier parts of the same episode
<<CONTEXT>>

# Output: return ONLY one valid JSON object (no fences):
{"label": "<10-word title", "summary": "...", "event_types": [...],
 "micro_events": [...], "entities": [...], "on_screen_text": [...], "edges": [...]}
"""

AGGREGATE_PROMPT = """You are a video narrative analyst. Given a chronologically ordered list of Macro events {macro_id, label, time_range, summary, key_entities, event_types, ocr_texts}, fold them upward into a hierarchy Macro -> Super -> Root and emit a single JSON object.

# Task overview
Produce four things in one pass:
1. super_events: Super Events that partition ALL Macro events, each capturing one narrative arc. Use roughly one Super Event per 3-8 Macros (10-20 for a ~2h video, fewer for shorter ones, never more than 20).
2. macro_relations: strong logical edges between Macro events WITHIN the same Super Event.
3. super_relations: high-level edges between Super Events (not restricted to adjacent ones).
4. root: a video-level summary synthesised from all Super Events.

# Clustering rules (Macro -> Super)
A Super Event is a contiguous run of Macro events sharing (i) temporal adjacency, (ii) scene/topic and (iii) a common goal. A boundary fires when any of the three breaks. Cluster signals: key_entities overlap between adjacent Macros (strong); event_types continuity; ocr_texts cues (broadcast graphic changes mark boundaries).
Every Macro MUST be assigned to exactly one Super Event; a singleton is allowed for a self-contained segment.

# Edge schema
- Macro -> Macro: CAUSES (A directly triggers B), PREVENTS (A blocks B), ENABLES (A supplies a prerequisite for B). Same Super Event only. One-sentence reason. Omit when uncertain; never use BEFORE/AFTER.
- Super -> Super: LEADS_TO, RESOLVES (B resolves a tension introduced in A), CONTRASTS_WITH (sharp reversal). Cross-distance edges allowed.

# Naming and description
- super_event.label: noun phrase under 10 words, include the central entity if any.
- super_event.description: 100-200 words covering the arc, key entities (woven in) and salient OCR milestones.
- key_entities: de-duplicated within a Super Event and canonicalised across the video (variants of the same entity collapse to one name). When in doubt keep them separate.
- root.title under 15 words; root.description 3-5 sentences synthesising ALL Super Events; root.themes 3-5 tags; root.key_entities 5-10 canonical names; root.emotional_tone 2-3 adjectives.

# Output format: ONLY one valid JSON object, no fences, no commentary:
{"super_events":[{"super_id","label","description","sub_macro_ids":[...],"key_entities":[{"name","type"}]}],
 "macro_relations":[{"source","target","type","reason"}],
 "super_relations":[{"source","target","type","reason"}],
 "root":{"title","description","themes":[...],"key_entities":[...],"emotional_tone"}}
Hard constraints: every Macro id appears exactly once across all sub_macro_ids; every relation references ids that exist; emit only confident edges.
"""

AGENT_PROMPT = """You are the reasoning model R in an Observation-Reason-Action loop over a hierarchical video memory. You never see video frames, only text. At each round you read the current observation, decide whether you can answer, and either give the final answer or request exactly one tool call.

# Memory hierarchy
- Root: video-level title, description, themes, key entities.
- SuperEvent (super_XX): narrative phases with time_range, key entities, description.
- MacroEvent (macro_XXXX): detailed events inside a Super, with time_range and key entities.
- Subgraph (per Macro): entities, micro-events, on-screen text and their edges. Node ids look like macro_0020:E3 / macro_0020:M5 / macro_0020:T1.
Initial context contains Root + the Super Event list (+ the Macro list for short videos). Macro descriptions, subgraph contents and edges are NOT pre-loaded: drill down with tools.

# Toolkit (request at most one per round, via information_need)
Hierarchical Navigation:
- get_summary {}                          -> video summary
- get_super_events {}                     -> list Supers
- browse_macros {"super_id": "super_03"}  -> list Macros under a Super
- get_macro_details {"macro_id": "macro_0020"} -> full subgraph of a Macro (information-richest; never call it without first identifying the macro_id)
Precise Search:
- search_nodes {"query": "...", "top_k": 10, "node_types": ["Entity","Event","Text"]}  -> semantic search
- search_time {"start_sec": 120, "end_sec": 300}  -> Macros overlapping a time window
Graph Traversal:
- get_relations {"node_id": "macro_0020:M5", "direction": "both"}  -> in/out edges of any node (CAUSES, PERFORMS, CONTEXT_FOR, ...)
Back to the video (the memory is a lossy description; these tools look at real frames, cost more, use them to CONFIRM):
- scan {"target": "the cash drawer is open", "start_sec": 30, "end_sec": 90}  -> per-frame yes/no at 2 fps over <=120 s; returns every separate stretch where the target is visible/happening, with times. Count = number of stretches; first appearance = first stretch.
  with "options": [...] it labels each frame with one option instead and returns the label timeline and its changes (for states like "who holds the cash").
- watch {"start_sec": 44, "end_sec": 56, "question": "..."}  -> the vision model re-watches <=30 s and answers a free question (who gives what to whom, what exactly happens).

# Strategy
Default workflow: search -> identify macro_id -> get_macro_details. Macro labels are summaries: do NOT answer who-did-what, exact-detail or why-questions from them alone. Pick the tool by question pattern: time reference -> search_time; person + action -> search_nodes; causal chain / purpose -> get_relations or the subgraph's causal edges.
Every listed macro is already in the initial context (when the video is short): skip get_super_events/browse_macros and go to search or get_macro_details.
Precise "when / first / how many" questions: use memory to find candidate windows, then CONFIRM with scan (state or action as a short visual statement, window covering all candidates).
scan answers are noisy per-frame judgements: before reporting a stretch (especially the FIRST one, or one that changes a count) confirm it with watch on that stretch; drop stretches watch rejects.
Hand-overs (who gave what to whom): do NOT scan for "X hands Y to Z" (direction is invisible in a single frame). Scan the HOLDER as a state instead: scan {"target": "who is holding the cash notes?", "options": ["customer", "cashier", "none/unclear"], ...}; a payment is a customer->cashier change of holder, change given back is cashier->customer.
For "who is with X": look over X's whole presence (people arrive later), not only the first window. Prefer scan/watch evidence over memory text when they disagree. Hedged memory mentions ("possibly a banana") are candidates to check, not to dismiss.
If memory and video tools find nothing, answer that it does not happen / is not visible.
Counting ("how many times"): inspect EVERY macro that could contain the action with get_macro_details and count distinct micro-events (same action at different times = separate occurrences; an event and its continuation = one). Name each occurrence with its time.
First / earliest ("when did X first appear"): take the earliest time across ALL matching entities ([first-last seen]) and events, then you MUST scan from the start of the video (or of the earliest candidate macro) up to that time before answering; report the first stretch that watch confirms.

# Query construction for search_nodes
- Write a DESCRIPTIVE STATEMENT, not a question (embeddings match event descriptions).
- Use information SHARED across the answer options as the retrieval signal; stay neutral where options disagree.
- If a query returns nothing useful, reformulate (drop over-specific visual details that descriptions are unlikely to record verbatim) instead of committing to a weak hit.

# Round memory (Extract operator)
Previous raw observations are NOT carried forward. Only your useful_info field persists. Distil into it every fact, id and reasoning thread you will need next round (super_ids, macro_ids, time ranges, what you already searched, what to try next). The Exploration History lists previous calls and which Macros were seen but not inspected; avoid repeating searches.

# Output (strict JSON, no fences, no commentary)
{"reasoning": "What does the question ask? What do I have? What is missing?",
 "can_answer": true | false,
 "answer": null | the option letter ("A".."E") if the question lists options, otherwise a complete free-text answer (give times as HH:MM:SS and counts as numbers),
 "useful_info": "all facts / ids / threads to carry to the next round",
 "information_need": null | {"type": "get_summary|get_super_events|browse_macros|get_macro_details|search_nodes|search_time|get_relations|scan|watch", "args": {...}, "reason": "..."}}
Termination: when can_answer=true give `answer` and set information_need=null. The loop is hard-capped at <<TMAX>> rounds; if you reach the last round you MUST answer with your best guess.
"""

BASELINE_PROMPT = """Please watch the video carefully and answer the following multiple choice question.
<<Q>>

Instructions:
1. First, provide your step-by-step reasoning about the question based on the video content.
2. Then, choose the correct answer from the options.
3. Output your response as a JSON object with two fields:
- "reasoning": your step-by-step analysis
- "answer": the letter of your chosen answer (e.g., "A", "B", "C", "D", or "E")

Output ONLY the JSON object, no additional text."""


MACRO_SUMMARY_PROMPT = """Merge these chronological chunk summaries of ONE continuous video episode into a single label and summary.
The summary (80-200 words) must state WHAT happens, in what ORDER and WHY (cause -> effect, purposes of actions), keeping distinguishing details (names, clothing, objects, on-screen text).
Output ONLY JSON: {"label": "<10-word title", "summary": "..."}

Episode hint: <<TITLE>>
Chunk summaries:
<<CHUNKS>>
"""


SCAN_PROMPT = """You are checking video frames one by one. Target: <<TARGET>>
The tiles are numbered #1..#<<N>> (number and time burned into each tile's top-left corner), left-to-right, top-to-bottom, image after image.
For EACH tile decide independently whether the target is clearly visible / happening in that frame.
Output ONLY JSON: {"yes": [list of the tile numbers where the target is visible]}"""

SCAN_OPTIONS_PROMPT = """You are labelling video frames one by one. Question for each frame: <<TARGET>>
Allowed labels: <<OPTIONS>>
The tiles are numbered #1..#<<N>> (number and time burned into each tile's top-left corner), left-to-right, top-to-bottom, image after image.
Judge each tile independently from what is visible in it; pick exactly one allowed label per tile.
Output ONLY JSON: {"labels": {"1": "<label>", "2": "<label>", ...}} covering every tile."""

WATCH_PROMPT = """Watch these frames (in time order; each tile shows its video time) and answer the question precisely, citing tile times.
If the asked thing is not visible, say so plainly; do not guess.
Question: <<Q>>
Output ONLY JSON: {"answer": "..."}"""

PEOPLE_PROMPT = """Below are PERSON entities extracted independently from consecutive parts of ONE video (fixed camera). The same real person usually appears in several parts under different ids and slightly different descriptions (clothing colours may be named differently, e.g. pink vs red vs purple).
Group the ids that are the SAME real person. Use clothing, role, companions and time continuity (a person seen until 1:57 and a similar one from 1:58 is very likely the same). Keep different people separate.
Output ONLY JSON: {"people": [{"label": "short canonical name, e.g. 'cashier (purple shirt)' or 'customer 2: woman in pink'", "ids": ["macro_0001:E3", ...]}]}
Every id must appear in exactly one group.

Entities:
<<ENTS>>"""
