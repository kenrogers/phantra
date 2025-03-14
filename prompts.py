from langchain_core.messages import SystemMessage

# Voice analysis prompt - Phase One
VOICE_ANALYSIS_PROMPT = SystemMessage(content="""
Analyze the speaking style and voice characteristics in the transcript, focusing on micro, meso, and macro-level elements.

MICRO-LEVEL ELEMENTS:
1. Vocabulary and word choice
   - Analyze unique, specialized, or repetitive words and phrases
   - Identify frequency and diversity of vocabulary
   - Examine use of jargon, slang, or colloquialisms
   - Determine preference for simple or complex words
   - Note distinctive or unusual word choices
   - Assess reading level required to understand the vocabulary

2. Grammatical patterns
   - Identify specific grammatical structures (passive voice, complex tenses, parts of speech)
   - Analyze verb tense usage and consistency
   - Examine use of singular/plural nouns and pronouns
   - Note recurring grammatical patterns or deliberate deviations

3. Punctuation
   - Analyze use of punctuation marks (commas, semicolons, dashes, parentheses)
   - Determine if certain punctuation marks are favored
   - Examine how punctuation creates rhythm, emphasis, or clarity
   - Identify unusual or idiosyncratic punctuation patterns

MESO-LEVEL ELEMENTS:
4. Sentence structure and length
   - Examine complexity, variety, and average length of sentences
   - Identify use of simple, compound, complex, or compound-complex sentences
   - Analyze use of sentence fragments or varied sentence beginnings
   - Determine preference for short, punchy sentences or longer, elaborate ones
   - Note average words per sentence and distribution of sentence lengths

5. Rhetorical devices
   - Recognize use of literary techniques (metaphors, similes, alliteration, repetition)
   - Identify rhetorical questions, irony, or hyperbole
   - Examine use of analogies, anecdotes, or examples to illustrate points
   - Note distinctive or recurring rhetorical devices

6. Paragraph structure
   - Examine organization of thoughts, including length and transitions
   - Analyze use of deductive or inductive reasoning
   - Identify chronological, spatial, or emphatic ordering
   - Note unique or recurring patterns in thought structure

MACRO-LEVEL ELEMENTS:
7. Tone and mood
   - Analyze overall emotional tone (formal, casual, humorous, serious)
   - Identify tone shifts and how they're achieved
   - Determine how diction, syntax, and imagery create specific mood
   - Note any inconsistencies or contradictions in tone

8. Overall coherence and cohesion
   - Analyze how ideas flow and connect throughout
   - Examine use of transitional words, phrases, or sentences
   - Identify use of repetition, parallel structure, or other cohesive devices
   - Determine ability to maintain focus and unity

9. Idiosyncrasies and quirks
   - Identify unique aspects of speaking style (unconventional phrases, made-up words)
   - Note recurring stylistic quirks or mannerisms
   - Analyze how these idiosyncrasies contribute to effectiveness

10. Figurative language
    - Identify use of personification, synecdoche, metonymy, etc.
    - Analyze how figurative language conveys meaning or creates imagery
    - Examine frequency and effectiveness of figurative language

For each element, provide specific examples from the transcript that illustrate these characteristics.
""")

# Voice analysis prompt - Phase Two
VOICE_ANALYSIS_PHASE_TWO_PROMPT = SystemMessage(content="""
Based on the detailed voice analysis provided, create comprehensive guidelines for emulating this speaking style. These guidelines will be used by a writer to create content that authentically matches the original speaker's voice.

Your guidelines should:

1. Distill the key characteristics of the speaker's voice into clear, actionable instructions
2. Provide specific examples of what to do and what to avoid
3. Include templates or patterns that capture the speaker's typical sentence structures
4. Highlight distinctive phrases, transitions, or expressions to incorporate
5. Explain how to handle technical concepts in the speaker's style
6. Detail how to maintain the speaker's tone and level of formality
7. Describe how to structure paragraphs and arguments as the speaker would

Format your response as a detailed style guide with clear sections for vocabulary, grammar, sentence structure, rhetorical devices, tone, and any other relevant aspects of the speaker's voice. Include "do this" and "don't do this" examples for each section.

This guide will be directly used by the writer to create social posts that sound authentically like the original speaker.
""")

# Insight extraction prompt
INSIGHT_EXTRACTION_PROMPT = SystemMessage(content="""
Extract exactly 7 standalone, highly shareable insights from this YouTube transcript, focusing on:

1. CONCRETE EXAMPLES: Identify specific, actionable examples the speaker provides that demonstrate their main concepts (e.g., "The speaker demonstrates how 'vibe coding' involves creating a focused environment with specific music and lighting before starting coding sessions")

2. UNIQUE PERSPECTIVES: Capture novel viewpoints or counterintuitive ideas that challenge conventional thinking in this domain

3. PRACTICAL TAKEAWAYS: Extract actionable advice or techniques viewers can immediately implement

4. MEMORABLE QUOTES: Identify powerful, concise statements that encapsulate a key idea (include timestamp)

5. STATISTICAL INSIGHTS: Note any data points, research findings, or quantifiable information

6. PROBLEM-SOLUTION PAIRS: Identify specific problems mentioned and their corresponding solutions

7. CONCEPTUAL FRAMEWORKS: Extract any mental models, frameworks, or systematic approaches described

For each insight:
- Write a 1-2 sentence summary of the insight
- Include relevant timestamps (format: [HH:MM:SS])
- Add a brief explanation of why this insight is valuable to the target audience
- Rate shareability potential from 1-5 (5 being highest)

FORMAT EACH INSIGHT AS:
Summary: [1-2 sentence distillation of the key insight]
Timestamp: [HH:MM:SS]
Value: [Why this matters to the audience]
Shareability: [1-5]
""")

# LinkedIn post writer prompt
LINKEDIN_POST_WRITER_PROMPT = SystemMessage(content="""
Create a high-performing LinkedIn post for each extracted insight that authentically matches the original speaker's voice.

## CONTENT REQUIREMENTS
1. Use the extracted insight as the foundation for your post
2. Incorporate the speaker's voice characteristics from the provided style guide
3. Keep the post under 3,000 characters (aim for 1,300-2,000 for optimal engagement)
4. Create plain text posts (no fancy formatting required)
5. Never use hashtags

## POST STRUCTURE
1. Start with an attention-grabbing headline based on the insight
2. Include 3-4 hard paragraph breaks between headline and opening to create intrigue
3. Break text into single-sentence paragraphs for easy skimming
4. Use strategic spacing to improve readability
5. Include a specific call-to-action that encourages engagement
6. End with a thoughtful question related to the insight

## HEADLINE APPROACHES (Choose the most appropriate for each insight)
- Compelling statistics from the insight
- Direct quote from the speaker (use the timestamp to find exact wording)
- Thought-provoking question related to the insight
- "How-to" offering based on practical takeaways
- Bold/controversial statement that challenges conventional thinking
- Intriguing hook that creates curiosity (avoid clickbait)

## CONTENT STRATEGIES
1. For CONCRETE EXAMPLES:
   - Highlight the specific, actionable example
   - Explain how it demonstrates the concept
   - Show how readers can apply it themselves

2. For UNIQUE PERSPECTIVES:
   - Present the counterintuitive idea clearly
   - Explain why it challenges conventional thinking
   - Provide the speaker's reasoning or evidence

3. For PRACTICAL TAKEAWAYS:
   - Present as step-by-step advice
   - Make it immediately actionable
   - Explain the benefit of implementation

4. For MEMORABLE QUOTES:
   - Feature the quote prominently
   - Provide context and explanation
   - Connect it to reader's experience

5. For STATISTICAL INSIGHTS:
   - Highlight the data point or research finding
   - Explain its significance
   - Connect it to practical application

6. For PROBLEM-SOLUTION PAIRS:
   - Clearly identify the problem
   - Present the speaker's solution
   - Explain why this solution is effective

7. For CONCEPTUAL FRAMEWORKS:
   - Outline the framework or mental model
   - Explain how it works
   - Show how it can be applied

## VOICE AUTHENTICITY
- Use vocabulary, sentence structures, and rhetorical devices that match the speaker's style
- Maintain the speaker's tone and level of formality
- Incorporate distinctive phrases or expressions identified in the style guide
- Handle technical concepts in the way the speaker would

For each post, provide:
1. The complete LinkedIn post with proper spacing and formatting
2. A brief explanation of your strategic approach
""")
