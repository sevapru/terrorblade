Analyze this dialogue and create:

1. **Structured summary** with header and subheaders by discussion parts
2. **Extract unique entities** (games/apps/brands/names/places/events/slang/tech) with emotional analysis
3. **One-word tags** for themes and emotional tone dynamics between participants
4. **Reference examples** with short quotes where key unique entities appear

## Language and Style
- Use the original chat's language. Preserve conversation tone and vocabulary (direct, no bureaucratic language)
- Be concise and to the point; no fluff

## Analysis Steps

### 1. Conversation Structure:
- **Events Timeline**: Create a chronological diagram of key events (format: Event → Consequence → Participant Reaction)
- Identify key topics and ideas
- Form header and subheaders by discussion logic
- Note cause-effect relationships and conflict points
- **Participant Score Tracking**: Calculate dynamic score changes for each participant throughout dialogue based on their emotional state, social interactions, and outcomes

### 2. Extract Unique Entities (UniqueTags):
- Only what's explicitly present in text
- Categories: Games, Apps/Services, Brands/Objects, People, Places, Events, Slang/Memes, Tech/Household
- Selection criteria: proper/named entities, narrow/rare terms, brands/models/games/toponyms
- Normalization: token (lowercase, spaces→hyphens), display_name (exact form from chat)
- **Emotional Analysis**: For each entity add emotional_tone (positive/negative/neutral) and participant_attitude (how each participant relates to this entity)
- Include metadata: category, frequency, salience (1-5), first_seen_index, references (1-3 short quotes)
- Exclude generic words like "work", "relationships" - those go to ThemeTags
- Rank by salience then frequency
- Limit: up to 15 items

### 3. Thematic and Emotional Tone Analysis:
- ThemeTags: 5-10 key themes, one word, lowercase
- **EmotionalDynamics**: Track emotional pairs between participants throughout dialogue:
  - Initial emotional states of each participant
  - Emotional shifts and triggers
  - Final emotional states
  - Dynamic patterns (escalation/de-escalation/synchronization/conflict)
- ToneTags: 3-7 tonal tags
- All tags as single-word tokens (lowercase, spaces→hyphens)

## Output Format
1. Header
2. **Events Timeline** (Event → Impact → Participant Response)
3. Subheaders + brief points by parts
4. **Participant Score Evolution** (track changes from start to end)
5. One-word tags:
   - UniqueTags: [tokens...] with emotional_context for each
   - ThemeTags: [tokens...]
   - **EmotionalDynamics**: [participant_pair: initial_state → shifts → final_state]
   - ToneTags: [tokens...]
6. Brief conclusions (2-4 lines)

## Conversation Data:
Chat: {group['chat_name']}
Messages: {group['message_count']}
Words: {group['total_words']:,}
Participants: {group['participants']}
Avg: {group['avg_words_per_message']:.1f} words/msg

## Messages:
{chr(10).join(messages_text)}

Analyze maintaining the original language and tone. Focus on accurate event interpretation - avoid labeling normal social situations as "criminal". If no unique entities found, return empty UniqueTags list.