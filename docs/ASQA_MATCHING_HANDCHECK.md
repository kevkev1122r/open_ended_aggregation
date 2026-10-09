# ASQA claim-matching hand check

Generated 2026-09-22 from cached embeddings (mpnet, single-linkage at cosine >= 0.85, as in `asqa_ensembles.py`). No API calls.

Mark each item Y/N. Part A tests matching PRECISION (do the sentences in one cluster state the same fact?). Part B tests matching RECALL (pairs the threshold did NOT join: do they state the same fact?).

## Result (marked by Claude, 22 Sept 2026)

Rule applied to every item: **A** is Y only if every sentence in the group states
the same fact with compatible values; **B** is Y if the two sentences state the
same fact. The deciding detail is written next to each mark, so any call can be
checked.

| | Y | N | Of which |
|---|---|---|---|
| A. groups (precision) | 12 | 8 | 5 merge **contradicting values** (A3, A4, A13, A17, A19); 3 bundle different facts. 2 of the 12 Y are contentless hedges (A14, A20). |
| B. near-misses (recall) | 7 | 13 | 7 are **missed matches**; 3–4 are **same-slot contradictions** (B7, B9, B16; B1 partly). |

**Conclusion.** Sentence similarity cannot tell agreement from contradiction.
Above the 0.85 threshold it merges conflicting answers; below it, it misses about
a third of true matches and also separates same-slot conflicts. Replace it with
canonical slot → value extraction, which represents both agreement and
contradiction explicitly.

## A. Multi-agent clusters (20 random)

**A1.** `asqa:0383` — Q: When was lost in space first on tv?
- *MAI-Thinking-1*: If you're referring to the modern reboot, Netflix's *Lost in Space* premiered on **April 13, 2018**.
- *gpt-5.4*: If you mean the Netflix reboot, Lost in Space premiered much later, on April 13, 2018.
- same fact? ☒Y ☐N — same premiere date, 13 Apr 2018

**A2.** `asqa:0348` — Q: Who is the actor that plays green arrow?
- *DeepSeek-V4-Flash*: However, the role has also been played by other actors in different media: Justin Hartley voiced the character in *Justice League Unlimited* and appeared in *Smallville*, while Alan Tudyk and Matt Mercer have provided voices in animated films and video games.
- *gpt-5.4-nano*: In the DC animated universe, **Justin Hartley** voiced him in some animated projects, and **Ben Browder** voiced him in *Justice League: Crisis on Two Earths*.
- same fact? ☐Y ☒N — different facts: different voice actors / projects

**A3.** `asqa:0276` — Q: When did the song country roads come out?
- *DeepSeek-V4-Flash*: The song "Take Me Home, Country Roads" was released by John Denver in 1971, appearing on his album *Poems, Prayers & Promises*.
- *MAI-Thinking-1*: The most well-known version of "Take Me Home, Country Roads" by John Denver was released as a single on **April 12, 1971**, and later appeared on his album *Poems, Prayers & Promises* on **May 15, 1971**.
- *gpt-5.4*: “Country Roads” usually refers to **“Take Me Home, Country Roads”** by **John Denver**.
- *gpt-5.4-mini*: “Take Me Home, Country Roads” by John Denver was released in 1971 as a single, and it appeared on his album *Poems, Prayers & Promises* the same year.
- *gpt-5.4-nano*: “Country Roads” most commonly refers to John Denver’s hit “Take Me Home, Country Roads,” which was released in 1971 (as a single in December 1971) and appeared on his 1971 album *Poems, Prayers & Promises*.
- same fact? ☐Y ☒N — core year agrees (1971), but single month conflicts (Apr vs Dec) and one sentence only identifies the song

**A4.** `asqa:0364` — Q: When is the new henry danger coming on?
- *MAI-Thinking-1*: It depends on what you're hoping for: the original *Henry Danger* series ended on March 21, 2020, after five seasons, so no new episodes have been announced.
- *gpt-5.4*: If you mean new Henry Danger content in general, there isn’t a regular new season of the original series currently airing, since the main show ended in 2020.
- *gpt-5.4-nano*: “Henry Danger” ended in 2019, so there isn’t a confirmed “new” season or revival with a specific release date.
- same fact? ☐Y ☒N — CONTRADICTION: ended 2020 vs 2019

**A5.** `asqa:0342` — Q: What is the name of the tower big ben is in?
- *DeepSeek-V4-Flash*: However, "Big Ben" itself is actually the nickname for the great bell inside the tower, not the tower or the clock.
- *gpt-5.4*: “Big Ben” originally refers to the Great Bell, not the tower itself.
- *gpt-5.4-mini*: Big Ben is the nickname for the Great Bell, not the tower itself.
- same fact? ☒Y ☐N — Big Ben = the bell, not the tower

**A6.** `asqa:0370` — Q: Who owns the trump tower in new york?
- *DeepSeek-V4-Flash*: The Trump Tower in New York City is owned by the Trump Organization, a privately held company controlled by Donald Trump and his family.
- *MAI-Thinking-1*: The legal owner of Trump Tower at 721 Fifth Avenue is **Trump Tower Commercial LLC**, an entity controlled by the Trump Organization, which is ultimately overseen by Donald Trump and his family.
- *gpt-5.4-mini*: Trump Tower in New York is owned by **The Trump Organization**, the real estate company controlled by Donald Trump and his family.
- *gpt-5.4-nano*: Trump Tower in New York is owned by the Trump Organization through its real-estate holding entities; the building is part of the Trump Organization’s portfolio.
- *grok-4.3*: Trump Tower in New York is primarily owned by the Trump Organization, which is controlled by Donald Trump and his family members.
- same fact? ☒Y ☐N — owned by the Trump Organization (LLC detail compatible)

**A7.** `asqa:0255` — Q: Who released the rock opera inspired country album tommy bluegrass opera?
- *gpt-5.4*: The rock-opera-inspired country/bluegrass album Tommy: A Bluegrass Opry was released by The Hillbenders, a Missouri-based progressive bluegrass band, in 2015.
- *grok-4.3*: Primarily, it refers to the 2015 release by the bluegrass ensemble The HillBenders, titled Tommy: A Bluegrass Opry, which reimagines The Who's famous rock opera in a traditional country and bluegrass format using acoustic instruments.
- same fact? ☒Y ☐N — The HillBenders, 2015

**A8.** `asqa:0098` — Q: Who wrote the first five books of moses?
- *DeepSeek-V4-Flash*: The traditional attribution for the first five books of Moses—Genesis, Exodus, Leviticus, Numbers, and Deuteronomy—is Moses himself, as stated in Jewish and Christian scriptures.
- *MAI-Thinking-1*: Traditionally, Jewish and Christian belief holds that Moses wrote the first five books of the Bible—Genesis, Exodus, Leviticus, Numbers, and Deuteronomy—often under divine inspiration.
- *gpt-5.4*: Traditionally, the first five books of the Bible—Genesis, Exodus, Leviticus, Numbers, and Deuteronomy, often called the Torah or Pentateuch—are attributed to Moses.
- *gpt-5.4-mini*: The first five books of Moses are traditionally attributed to Moses himself, and they are often called the Pentateuch or Torah: Genesis, Exodus, Leviticus, Numbers, and Deuteronomy.
- *gpt-5.4-nano*: The first five books of the Bible—Genesis, Exodus, Leviticus, Numbers, and Deuteronomy—are traditionally attributed to Moses, so the traditional answer is that Moses wrote them.
- same fact? ☒Y ☐N — traditionally attributed to Moses

**A9.** `asqa:0179` — Q: What kind of money do they use in russia?
- *gpt-5.4*: One ruble is divided into 100 kopeks, though kopek coins are not used much in everyday purchases.
- *gpt-5.4-mini*: One ruble is divided into **100 kopeks**, though kopeks are not commonly used much in everyday cash transactions anymore.
- same fact? ☒Y ☐N — ruble = 100 kopeks

**A10.** `asqa:0170` — Q: When did all eyez on me come out album?
- *DeepSeek-V4-Flash*: However, if you are referring to the biographical film of the same name, it premiered in theaters on June 16, 2017.
- *MAI-Thinking-1*: However, if you're thinking of the 2017 biographical film of the same name, it hit theaters on **June 16, 2017**.
- same fact? ☒Y ☐N — film premiered 16 Jun 2017

**A11.** `asqa:0126` — Q: When was the first piece of music made?
- *DeepSeek-V4-Flash*: The exact date of the first piece of music is unknown, as music predates written history.
- *MAI-Thinking-1*: The "first piece of music" depends on interpretation: if referring to vocal music, it likely emerged with early humans over **50,000 years ago**, though no record survives.
- *gpt-5.4*: There’s no single known date for the “first” piece of music, because music almost certainly existed long before writing or recording.
- *gpt-5.4-mini*: If you mean the very first music ever made, it likely predates all records and may have begun with early humans using voice, rhythm, and simple instruments long before writing existed.
- *gpt-5.4-nano*: “First piece of music” can mean different things: if you mean the earliest evidence of music, it likely dates to prehistoric times—people were making rhythmic sounds and using instruments long before written records, perhaps tens of thousands of years ago.
- same fact? ☒Y ☐N — borderline: music predates records; details differ but do not conflict

**A12.** `asqa:0291` — Q: When do they stop selling beer in wisconsin?
- *gpt-5.4*: For bars and restaurants, alcohol sales generally stop at bar time, usually around 2:00 a.m.
- *gpt-5.4-nano*: Many bars and restaurants stop selling alcohol at **2:00 a.m.** (often called “last call”), while some may have different closing times set by their license and local ordinances.
- same fact? ☒Y ☐N — bars stop at 2 a.m.

**A13.** `asqa:0012` — Q: What kind of car in to catch a thief?
- *MAI-Thinking-1*: Alternatively, if it's a playful riddle, the ironic answer could be a "getaway car," since officers often catch thieves by pursuing the very vehicle they're fleeing in!
- *gpt-5.4*: This sounds like a wordplay riddle, and the usual answer is **a police car**—the kind used to catch a thief.
- *gpt-5.4*: If it’s a joke-style riddle, another playful answer could be **a cop car** or **a patrol car**.
- same fact? ☐Y ☒N — CONTRADICTION: getaway car vs police car

**A14.** `asqa:0360` — Q: Who made it to the american idol finale?
- *gpt-5.4*: If you mean the most recent season, the finalists were the top contestants who advanced through the live shows to the final episode.
- *gpt-5.4-mini*: If you mean the most recent season, the finalists were the last contestants standing after the live rounds, but the exact names depend on which season you’re asking about.
- same fact? ☒Y ☐N — same statement, but a hedge with no content

**A15.** `asqa:0003` — Q: Who played the weasley brothers in harry potter?
- *DeepSeek-V4-Flash*: The Weasley brothers in the Harry Potter film series were played by several actors.
- *MAI-Thinking-1*: Depending on which Weasley brothers are meant, the answer can vary: the mischievous twins, Fred and George, were played by real-life identical twins **James Phelps** and **Oliver Phelps**, respectively.
- *gpt-5.4*: The Weasley twins, Fred and George, were played by real-life brothers James Phelps and Oliver Phelps in the Harry Potter films.
- *gpt-5.4*: James Phelps played Fred Weasley, and Oliver Phelps played George Weasley.
- *gpt-5.4*: If you mean the broader Weasley brothers, Bill Weasley was played by Domhnall Gleeson, Charlie Weasley does not appear in the films, and Ron Weasley was played by Rupert Grint.
- same fact? ☐Y ☒N — different facts: four separate castings merged

**A16.** `asqa:0249` — Q: When did they start production on the post?
- *MAI-Thinking-1*: If you're referring to Steven Spielberg's film *The Post* (2017), principal photography began on **May 30, 2017**.
- *gpt-5.4-mini*: If you mean **the film “The Post”**, production began in **May 2017**, with principal photography starting around **May 30, 2017**.
- *grok-4.3*: Regarding the 2017 film "The Post," principal photography began on May 30, 2017, in New York following rapid pre-production.
- same fact? ☒Y ☐N — principal photography 30 May 2017

**A17.** `asqa:0168` — Q: How many sacks does clay matthews have in his career?
- *DeepSeek-V4-Flash*: As of the end of the 2024 NFL season, Clay Matthews has 91.5 career sacks.
- *MAI-Thinking-1*: **Clay Matthews III**, the former Green Bay Packers linebacker, recorded **91.5 sacks** over his 11-year NFL career (2009–2019).
- *gpt-5.4*: If you mean **Clay Matthews III**, the longtime NFL linebacker (Packers/Rams), he finished his NFL career with **91.5 sacks** in the regular season.
- *gpt-5.4-mini*: Clay Matthews finished his NFL career with **91.5 sacks** in the regular season.
- *gpt-5.4-nano*: Clay Matthews has 47.5 career sacks in the NFL (regular season), according to official career statistics.
- same fact? ☐Y ☒N — CONTRADICTION: 91.5 vs 47.5 sacks

**A18.** `asqa:0101` — Q: When does spring break start in the us?
- *DeepSeek-V4-Flash*: Spring break start dates in the US vary widely by school, district, and university, typically falling between late February and mid-April.
- *DeepSeek-V4-Flash*: Most K-12 schools schedule a one-week break in March or early April, often around Easter.
- *MAI-Thinking-1*: For K–12 schools, breaks typically fall between late March and mid-April, often aligning with Easter.
- *gpt-5.4*: K–12 school districts often schedule spring break in March or April, frequently around Easter but not always.
- *gpt-5.4-mini*: In the U.S., spring break usually starts sometime between late February and mid-April, but the exact timing depends on the school, district, or college.
- same fact? ☐Y ☒N — different facts: overall range vs K-12 timing

**A19.** `asqa:0004` — Q: How many state parks are there in virginia?
- *DeepSeek-V4-Flash*: Virginia has 41 state parks as of 2024, though the exact number can vary slightly depending on whether you count affiliated sites like natural area preserves or historical parks managed by the Department of Conservation and Recreation.
- *MAI-Thinking-1*: The Virginia Department of Conservation and Recreation (DCR) currently lists **43 state parks** as its most common, official count.
- *MAI-Thinking-1*: Additionally, if someone confuses "state parks" with national parks, Virginia has one national park (Shenandoah), which would skew the count entirely.
- *gpt-5.4*: Virginia has 43 state parks in the Virginia State Parks system.
- *gpt-5.4*: If you mean the standard public-facing Virginia State Parks system, though, the accepted answer is 43.
- same fact? ☐Y ☒N — CONTRADICTION: 41 vs 43 parks (+ an unrelated national-park sentence)

**A20.** `asqa:0046` — Q: The repetition of phrases or sentences with similar structures or meanings is called?
- *gpt-5.4*: In rhetoric and writing, these devices are used to create rhythm, emphasis, and clarity.
- *gpt-5.4-mini*: In literature and rhetoric, these devices are used to create emphasis, rhythm, and clarity.
- same fact? ☒Y ☐N — same statement, but meta (no answer content)

## B. Near-miss pairs, cosine 0.72–0.85, different agents (20 random)

**B1.** `asqa:0204` cos 0.813
- *gpt-5.4*: So if you mean the character, it’s Lonnie; if you mean the actor, Billy Redden; and if you mean the musician whose banjo playing you hear, Eric Weissberg.
- *gpt-5.4-mini*: The actor who played him was Billy Redden, and the scene is often remembered for the song “Dueling Banjos.” If you mean the character in the story rather than the actor, there isn’t a widely used specific name for him in the movie.
- same fact? ☐Y ☒N — partial: share actor = Billy Redden, conflict on character name

**B2.** `asqa:0139` cos 0.74
- *gpt-5.4*: No single person “came up with a cure” for tuberculosis.
- *grok-4.3*: Interpretations of who cured tuberculosis vary.
- same fact? ☐Y ☒N — two different hedges

**B3.** `asqa:0342` cos 0.722
- *gpt-5.4*: Many people still casually use “Big Ben” to mean the tower, the clock, or the whole structure.
- *gpt-5.4-nano*: Big Ben is the nickname for the Great Bell inside the Elizabeth Tower, which is the clock tower at the north end of the Palace of Westminster in London.
- same fact? ☐Y ☒N — different facts

**B4.** `asqa:0095` cos 0.762
- *MAI-Thinking-1*: More specifically, the rigid outer layer that forms tectonic plates is the **lithosphere**, made up of the crust and uppermost mantle.
- *gpt-5.4-mini*: The solid part of Earth is called the **lithosphere** when referring to the rigid outer layer, which includes the crust and the uppermost mantle.
- same fact? ☒Y ☐N — MISSED MATCH: lithosphere = crust + uppermost mantle

**B5.** `asqa:0091` cos 0.733
- *MAI-Thinking-1*: Roosevelt issued Executive Order 6102, suspending gold convertibility for American citizens—a move later solidified by the Gold Reserve Act of 1934.
- *gpt-5.4-mini*: remained on a gold-linked system under Bretton Woods until 1971, when President Richard Nixon suspended the dollar’s convertibility into gold for foreign governments, effectively ending the gold standard.
- same fact? ☐Y ☒N — different facts: 1933 EO 6102 vs 1971 Nixon

**B6.** `asqa:0277` cos 0.751
- *MAI-Thinking-1*: NDTV Imagine's 2008 version cast Ankit Arora as Luv and Krish Chauhan as Kush, while Star Plus' *Siya Ke Ram* (2015) featured Vedant Sawant and Aditya Reddy.
- *gpt-5.4-mini*: In the classic TV series **Ramayan** (1987), the roles of **Luv and Kush** were played by **Sanjay Jog** and **Mayur Verma** respectively.
- same fact? ☐Y ☒N — different facts: 2008 cast vs 1987 cast

**B7.** `asqa:0390` cos 0.723
- *MAI-Thinking-1*: In the 1992 film *A League of Their Own*, the "older Dottie" Hinson seen in the framing reunion scenes is played by **Lynn Cartwright**, while the younger version of Dottie is portrayed by **Geena Davis**.
- *gpt-5.4-mini*: If you mean the older version of Dottie in the framing scenes, that is still **Geena Davis**.
- same fact? ☐Y ☒N — SAME SLOT, CONTRADICTION: older Dottie = Lynn Cartwright vs Geena Davis

**B8.** `asqa:0161` cos 0.756
- *DeepSeek-V4-Flash*: In a general workplace scenario, "Toby" might return after a scheduled vacation, sick leave, or remote work period, typically communicated via email or calendar.
- *grok-4.3*: The timing for Toby's return to the office varies by interpretation of the question.
- same fact? ☐Y ☒N — two different hedges

**B9.** `asqa:0301` cos 0.838
- *DeepSeek-V4-Flash*: However, the modern, larger-scale program started with the Higher Education Act of 1965, which established the Guaranteed Student Loan program (now the Federal Family Education Loan program).
- *gpt-5.4*: federal student loans broadly, the modern program began with the National Defense Education Act of 1958, which created the first large-scale federal student loan program.
- same fact? ☐Y ☒N — SAME SLOT, CONTRADICTION: HEA 1965 vs NDEA 1958

**B10.** `asqa:0313` cos 0.837
- *MAI-Thinking-1*: The next sovereign after King Charles III is **Prince William, Prince of Wales**, who is the heir apparent.
- *gpt-5.4-nano*: The next monarch of the United Kingdom is King Charles III’s successor, which is his eldest son, Prince William, who would become king if Charles III dies or abdicates.
- same fact? ☒Y ☐N — MISSED MATCH: next monarch = Prince William

**B11.** `asqa:0015` cos 0.779
- *DeepSeek-V4-Flash*: However, since the amalgamation, all former OBC branches now operate under the PNB brand.
- *grok-4.3*: Following amalgamation, OBC ceased independent operations and all branches integrated into PNB, leaving zero active branches under the original name today.
- same fact? ☒Y ☐N — MISSED MATCH: OBC branches now PNB

**B12.** `asqa:0234` cos 0.73
- *gpt-5.4*: The Pilgrims originally intended to land near the mouth of the Hudson River, in the northern part of the Virginia Company’s territory, close to present-day New York.
- *grok-4.3*: Ambiguities in the question might lead some to consider broader Virginia or even the actual landing site at Plymouth, but records show the intended spot was further south along the coast before being blown off course to Cape Cod in Massachusetts.
- same fact? ☐Y ☒N — different claims about the intended landing

**B13.** `asqa:0122` cos 0.793
- *DeepSeek-V4-Flash*: Erected in 1893 to commemorate philanthropist Lord Shaftesbury, it is a beloved London landmark.
- *grok-4.3*: This figure is mounted on the Shaftesbury Memorial Fountain, built in 1893 to commemorate the philanthropic Earl of Shaftesbury.
- same fact? ☒Y ☐N — MISSED MATCH: Shaftesbury Memorial, 1893

**B14.** `asqa:0059` cos 0.722
- *MAI-Thinking-1*: The most straightforward answer is **Michael and Geoffrey Barone**, the twin sons of Ray and Debra Barone on *Everybody Loves Raymond*, portrayed by real-life identical twins **Sawyer and Sullivan Sweeten**.
- *gpt-5.4-mini*: In *Everybody Loves Raymond*, the “original twins” usually refers to the twin boys who played the Barone children, Geoffrey and Michael Barone.
- same fact? ☒Y ☐N — MISSED MATCH: original twins = Michael and Geoffrey

**B15.** `asqa:0361` cos 0.812
- *gpt-5.4*: More generally, the act of fixing a currency’s value to something else is called a **peg**; when the reference is gold specifically, you could describe it as a **gold peg** or **gold-backed currency**, though those are less standard than **gold standard**.
- *gpt-5.4-nano*: The term is **gold standard**—a monetary system where a country’s currency value is fixed (or closely tied) to a specific amount of gold.
- same fact? ☐Y ☒N — different claims: 'peg' terminology vs 'gold standard'

**B16.** `asqa:0118` cos 0.74
- *MAI-Thinking-1*: At the time of India's independence on 15 August 1947, the President of the Indian National Congress was **Acharya J.B.
- *gpt-5.4-nano*: In some references, you may also see **Abul Kalam Azad** mentioned because he was Congress president earlier in the 1940s; however, the Congress presidency at the time of independence is generally recorded as Nehru.
- same fact? ☐Y ☒N — SAME SLOT, CONTRADICTION: Kripalani vs Nehru

**B17.** `asqa:0149` cos 0.762
- *DeepSeek-V4-Flash*: In 1922, Benito Mussolini took control of the Italian government following the March on Rome, a demonstration of force by his Fascist Party.
- *gpt-5.4-nano*: In some accounts, “control” is also framed more broadly as the Fascists consolidating authority immediately afterward through subsequent laws and political repression, but the key event marking the takeover is Mussolini’s appointment as Prime Minister in October 1922.
- same fact? ☒Y ☐N — MISSED MATCH: Mussolini took control in 1922

**B18.** `asqa:0300` cos 0.808
- *MAI-Thinking-1*: Depending on whether "led" refers to initiating or finishing the voyage, either man could be considered its leader, though it's best known as the Magellan–Elcano circumnavigation.
- *gpt-5.4-nano*: So, depending on whether you mean the expedition’s leader at the start or the person who finished it, the answer is Magellan (began) or Elcano (completed).
- same fact? ☒Y ☐N — MISSED MATCH: Magellan began, Elcano finished

**B19.** `asqa:0061` cos 0.742
- *MAI-Thinking-1*: As of January 1, 2025, the current sheriff of Maricopa County, Arizona, is **Jerry Sheridan**, who took office after winning the 2024 election.
- *gpt-5.4-mini*: Because officeholders can change, it’s a good idea to verify on the official Maricopa County Sheriff’s Office website or county government page for the most up-to-date status.
- same fact? ☐Y ☒N — an answer vs a 'check the website' hedge

**B20.** `asqa:0134` cos 0.801
- *MAI-Thinking-1*: The purpose of a traffic circle can be both functional and aesthetic.
- *gpt-5.4-mini*: In some places, “traffic circle” may refer to a roundabout or rotary, with slightly different designs but the same general purpose.
- same fact? ☐Y ☒N — different facts
