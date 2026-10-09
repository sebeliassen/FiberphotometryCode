# PROJECT B: Syllable Chains — Background, Tasks, & Roadmap

## B1. Short Project Background
I have been exploring how individual syllables combine into recurring behavioural sequences, or chains. This lets us ask whether a mouse repeats a particular routine, how it moves between routines and whether it performs the same chain differently. It adds a level of cohort-agnostic analysis between individual actions and broader behavioural states.

## B2. Tasks Performed and Approaches Tried

### Looking into what comes after the syllables
The starting question was what we could learn by looking beyond individual syllables. I looked into language-inspired approaches and methods such as BehaVERT, uBAM and shMoSeq, with the aim of finding useful descriptions of behaviour over longer periods. 

I also looked into whether the mice we work with are “memoryless” by looking at how much recent behavioural history helps predict future behaviour. These ideas formed the basis of the earlier presentation to Ulrik, “What Comes After the Syllable?”.

I then focused on shMoSeq, developed by Weinreb and his colleagues, as a natural extension of kpms. In the configuration I tested, the broader states were difficult to interpret consistently across animals. This did not provide the shared behavioural description I had hoped for, so I decided to start with a more tangible intermediate level: recurring chains of actions.

### Starting with interpretable chains
Chains provide concrete routines that can be analysed qualitatively just like kpms. I first explored counting exact sequences of syllables (exact n-grams), but this separates almost identical routines and tends to be dominated by frequent repetitive patterns (like resting periods). I therefore worked further with seqNMF, which groups recurring sequences while allowing some variation in how they are performed.

Much of this work has involved labelling syllables and making videos and visualisations to check what the chains actually represent. Themes found include:
- rearing → sniffing → descending
- right turn → running along wall → right turn
- groom → rest → groom

The initial analyses of the amph data suggest differences in how wildtypes and DAT-KI mice organise their behaviour. KI mice spend more time in running-related chains and appear more likely to repeat them, while wildtypes use more stationary chains.

## B3. Plan for the Coming Months

I plan to divide the next analyses into two parts: how syllables and chains are organised in relation to one another (inter-chain), and how the individual sequences are performed (intra-chain).

### Inter-chain analysis of organisation between syllables and chains
- **Repeats:** How often does the same chain occur again immediately afterwards?
- **Continuations:** How long does a mouse remain in the same routine, and how can we distinguish continuation from a new repetition?
- **Transitions:** Which chain does the mouse move to next, and does this differ between groups or conditions?
- **Initiations:** How often does a series of repeated chains begin, and how does the first chain differ from the subsequent repetitions?

### Intra-chain analysis of execution within syllables and chains
- **Consistency:** How similarly does a mouse perform the same chain each time, and does this variability differ between groups or conditions?
- **Completion:** How often is a chain completed, and where does it stop when it ends early?
- **Variations:** Which syllables are often added, omitted or repeated, and do these variations follow recurring patterns?
- **Pacing:** Which parts of a chain take longer, and how much time do different groups spend on these actions?
- **Context-dependent movement:** Is the same syllable performed differently depending on where it occurs within a chain?

### Intra-syllable analysis using the analyses above
Similar analyses could also be applied within individual syllables by treating each syllable as a sequence of frames. This opens up another level of analysis: e.g. how syllables vary and whether these variations are by omitting, adding or repeating micromovements and also how these frames are paced across different groups.
