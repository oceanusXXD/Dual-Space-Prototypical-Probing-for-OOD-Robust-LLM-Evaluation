"""Frozen ASAP-AES Judge-input contract.

The competition TSV contains student responses and scores but not the task
materials needed to interpret those scores.  This module vendors the task
material used by the Judge, records the archival source checksum, and creates
one deterministic, label-free input per essay.  It is deliberately separate
from score mapping: changing a rubric or template changes the template hash
and invalidates every Judge-input hidden-state cache.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


ASAP_PROMPT_CATALOG_VERSION = "asap_aes_prompt_rubric_catalog_v1"
ASAP_JUDGE_TEMPLATE_VERSION = "asap_overall_quality_rubric_v1"
_ARCHIVAL_SOURCE_BASE = (
    "https://raw.githubusercontent.com/vasu5235/"
    "Kaggle-Automated-Essay-Checking-System/master/Essay_Set_Descriptions/"
)

# The source files are the public archival mirror of the competition's
# ``Essay Set #N--ReadMeFirst.docx`` files.  We retain their SHA-256 values so
# a later re-download can be checked rather than silently changing the task.
# ``judge_rubric`` is a compact, faithful operational rendering of the listed
# official rubric; it is intentionally bounded so that source passage + rubric
# + essay fit the protocol's fixed 2,048-token context.  The scored trait and
# raw-score formula are stated explicitly for prompts 2, 7, and 8.
ASAP_PROMPT_CATALOG: dict[int, dict[str, Any]] = {
    1: {
        "source_sha256": "cee14d17e9c1a4a9ebb23cd4b790f349c5d76e546eb7625accb6b6c2ea9bc176",
        "source_file": "Essay%20Set%20%231--ReadMeFirst.docx",
        "task": (
            "More and more people use computers, but not everyone agrees that this benefits society. "
            "Those who support advances in technology believe that computers have a positive effect on people. "
            "They teach hand-eye coordination, give people the ability to learn about faraway places and people, "
            "and even allow people to talk online with other people. Others have different ideas. Some experts are "
            "concerned that people are spending too much time on their computers and less time exercising, enjoying "
            "nature, and interacting with family and friends.\n\nWrite a letter to your local newspaper in which you "
            "state your opinion on the effects computers have on people. Persuade the readers to agree with you."
        ),
        "source_passage": "",
        "judge_rubric": (
            "Official 1–6 holistic persuasive-writing rubric. Score 1: undeveloped, minimal/vague support, "
            "fragmented and hard to read. Score 2: under-developed, only general or list-like reasons, little "
            "organization. Score 3: minimally developed position with inadequate support; some organization. "
            "Score 4: clear position with adequate, partly specific support and satisfactory organization. "
            "Score 5: developed, clear position with mostly specific persuasive support, strong organization and "
            "transitions. Score 6: thoughtful clear position, fully elaborated specific support, strong organization, "
            "fluency, sophisticated transitions, and audience awareness."
        ),
        "scored_construct": "overall persuasive response quality",
        "raw_score_contract": "Each rater scores 1–6; resolved score is 2–12.",
    },
    2: {
        "source_sha256": "d5d56695814d48586e6622fe08da494d86df9ec2e8c90781bc55f832a0c439e3",
        "source_file": "Essay%20Set%20%232--ReadMeFirst.docx",
        "task": (
            "Censorship in the Libraries\n\n\"All of us can think of a book that we hope none of our children or "
            "any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- "
            "that work I abhor -- then you also have exactly the same right and so does everyone else. And then we "
            "have no books left on the shelf for any of us.\" -- Katherine Paterson, Author\n\nWrite a persuasive "
            "essay to a newspaper reflecting your views on censorship in libraries. Do you believe that certain "
            "materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they "
            "are found offensive? Support your position with convincing arguments from your own experience, "
            "observations, and/or reading."
        ),
        "source_passage": "",
        "judge_rubric": (
            "Use only Domain 1 (Writing Applications), the label present in this experiment. Official 1–6 holistic "
            "criteria assess task fulfillment and focused theme; relevant, developed ideas and support; coherent "
            "beginning/middle/end and transitions; precise vocabulary, technique and sentence fluency; and voice "
            "appropriate to task/audience. Score 6 is thorough, insightful and distinctive; 5 is solid and fully "
            "accomplished but less sophisticated; 4 is good with adequate development; 3 minimally accomplishes the "
            "task; 2 only partly accomplishes it with weak focus/development; 1 fails the task and is often too brief, "
            "rambling, repetitive, or difficult to understand. Do not score the separate Language Conventions domain."
        ),
        "scored_construct": "Domain 1 Writing Applications only",
        "raw_score_contract": "Domain 1 rater and resolved scores are each 1–6.",
    },
    3: {
        "source_sha256": "6dc0c142422171b589e9308e2ae9121d7337595cf08d489a960ac154470e4bc8",
        "source_file": "Essay%20Set%20%233--ReadMeFirst.docx",
        "task": (
            "Write a response that explains how the features of the setting affect the cyclist. In your response, "
            "include examples from the essay that support your conclusion."
        ),
        "source_passage": (
            "ROUGH ROAD AHEAD: Do Not Exceed Posted Speed Limit by Joe Kurmaskie\n\n"
            "FORGET THAT OLD SAYING ABOUT NEVER taking candy from strangers. No, a better piece of advice for the "
            "solo cyclist would be, ‘Never accept travel advice from a collection of old-timers who haven’t left the "
            "confines of their porches since Carter was in office.’ It’s not that a group of old guys doesn’t know the "
            "terrain. With age comes wisdom and all that, but the world is a fluid place. Things change.\n\n"
            "At a reservoir campground outside of Lodi, California, I enjoyed the serenity of an early-summer evening "
            "and some lively conversation with these old codgers. What I shouldn’t have done was let them have a peek "
            "at my map. Like a foolish youth, the next morning I followed their advice and launched out at first light "
            "along a ‘shortcut’ that was to slice away hours from my ride to Yosemite National Park.\n\n"
            "They’d sounded so sure of themselves when pointing out landmarks and spouting off towns I would come to "
            "along this breezy jaunt. Things began well enough. I rode into the morning with strong legs and a smile on "
            "my face. About forty miles into the pedal, I arrived at the first ‘town.’ This place might have been a "
            "thriving little spot at one time—say, before the last world war—but on that morning it fit the traditional "
            "definition of a ghost town. I chuckled, checked my water supply, and moved on. The sun was beginning to "
            "beat down, but I barely noticed it. The cool pines and rushing rivers of Yosemite had my name written all "
            "over them.\n\n"
            "Twenty miles up the road, I came to a fork of sorts. One ramshackle shed, several rusty pumps, and a corral "
            "that couldn’t hold in the lamest mule greeted me. This sight was troubling. I had been hitting my water "
            "bottles pretty regularly, and I was traveling through the high deserts of California in June.\n\n"
            "I got down on my hands and knees, working the handle of the rusted water pump with all my strength. A "
            "tarlike substance oozed out, followed by brackish water feeling somewhere in the neighborhood of two "
            "hundred degrees. I pumped that handle for several minutes, but the water wouldn’t cool down. It didn’t "
            "matter. When I tried a drop or two, it had the flavor of battery acid.\n\n"
            "The old guys had sworn the next town was only eighteen miles down the road. I could make that! I would "
            "conserve my water and go inward for an hour or so—a test of my inner spirit.\n\n"
            "Not two miles into this next section of the ride, I noticed the terrain changing. Flat road was replaced "
            "by short, rolling hills. After I had crested the first few of these, a large highway sign jumped out at "
            "me. It read: ROUGH ROAD AHEAD: DO NOT EXCEED POSTED SPEED LIMIT.\n\n"
            "The speed limit was 55 mph. I was doing a water-depleting 12 mph. Sometimes life can feel so cruel.\n\n"
            "I toiled on. At some point, tumbleweeds crossed my path and a ridiculously large snake—it really did look "
            "like a diamondback—blocked the majority of the pavement in front of me. I eased past, trying to keep my "
            "balance in my dehydrated state.\n\n"
            "The water bottles contained only a few tantalizing sips. Wide rings of dried sweat circled my shirt, and "
            "the growing realization that I could drop from heatstroke on a gorgeous day in June simply because I "
            "listened to some gentlemen who hadn’t been off their porch in decades, caused me to laugh.\n\n"
            "It was a sad, hopeless laugh, mind you, but at least I still had the energy to feel sorry for myself. "
            "There was no one in sight, not a building, car, or structure of any kind. I began breaking the ride down "
            "into distances I could see on the horizon, telling myself that if I could make it that far, I’d be fine.\n\n"
            "Over one long, crippling hill, a building came into view. I wiped the sweat from my eyes to make sure it "
            "wasn’t a mirage, and tried not to get too excited. With what I believed was my last burst of energy, I "
            "maneuvered down the hill.\n\n"
            "In an ironic twist that should please all sadists reading this, the building—abandoned years earlier, by "
            "the looks of it—had been a Welch’s Grape Juice factory and bottling plant. A sandblasted picture of a young "
            "boy pouring a refreshing glass of juice into his mouth could still be seen.\n\n"
            "I hung my head. That smoky blues tune ‘Summertime’ rattled around in the dry honeycombs of my deteriorating "
            "brain.\n\n"
            "I got back on the bike, but not before I gathered up a few pebbles and stuck them in my mouth. I’d read "
            "once that sucking on stones helps take your mind off thirst by allowing what spit you have left to circulate. "
            "With any luck I’d hit a bump and lodge one in my throat.\n\n"
            "It didn’t really matter. I was going to die and the birds would pick me clean, leaving only some expensive "
            "outdoor gear and a diary with the last entry in praise of old men, their wisdom, and their keen sense of "
            "direction. I made a mental note to change that paragraph if it looked like I was going to lose "
            "consciousness for the last time.\n\n"
            "Somehow, I climbed away from the abandoned factory of juices and dreams, slowly gaining elevation while "
            "losing hope. Then, as easily as rounding a bend, my troubles, thirst, and fear were all behind me.\n\n"
            "GARY AND WILBER’S FISH CAMP—IF YOU WANT BAIT FOR THE BIG ONES, WE’RE YOUR BEST BET!\n\n"
            "‘And the only bet,’ I remember thinking.\n\n"
            "As I stumbled into a rather modern bathroom and drank deeply from the sink, I had an overwhelming urge to "
            "seek out Gary and Wilber, kiss them, and buy some bait—any bait, even though I didn’t own a rod or reel.\n\n"
            "An old guy sitting in a chair under some shade nodded in my direction. Cool water dripped from my head as I "
            "slumped against the wall beside him. ‘Where you headed in such a hurry?’ ‘Yosemite,’ I whispered. ‘Know "
            "the best way to get there?’ I watched him from the corner of my eye for a long moment. He was even older "
            "than the group I’d listened to in Lodi. ‘Yes, sir! I own a very good map.’ And I promised myself right then "
            "that I’d always stick to it in the future."
        ),
        "judge_rubric": (
            "Official 0–3 source-dependent-response rubric. Score 3 demonstrates understanding of textual "
            "complexities, addresses the question, uses expressed and implied information, and extends beyond the "
            "literal. Score 2 is partial/literal: it addresses the question and uses some evidence but may not connect "
            "support to a conclusion. Score 1 is minimal understanding, may misread text/question, and lacks supporting "
            "explanation. Score 0 is irrelevant, incorrect, or no response."
        ),
        "scored_construct": "source-dependent reading response",
        "raw_score_contract": "Each rater and resolved score is 0–3.",
    },
    4: {
        "source_sha256": "d4158d1c1da4853f971ddc397151a4404c13de9bb7382ca271da4d122481292c",
        "source_file": "Essay%20Set%20%234--ReadMeFirst.docx",
        "task": (
            "Read the last paragraph of the story: ‘When they come back, Saeng vowed silently to herself, in the "
            "spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that "
            "test again.’ Write a response that explains why the author concludes the story with this paragraph. In "
            "your response, include details and examples from the story that support your ideas."
        ),
        "source_passage": (
            "Winter Hibiscus by Minfong Ho\n\nSaeng, a teenage girl, and her family have moved to the United States "
            "from Vietnam. As Saeng walks home after failing her driver’s test, she sees a familiar plant. Later, she "
            "goes to a florist shop to see if the plant can be purchased.\n\nIt was like walking into another world. A "
            "hot, moist world exploding with greenery. Huge flat leaves, delicate wisps of tendrils, ferns and fronds "
            "and vines of all shades and shapes grew in seemingly random profusion.\n\n‘Over there, in the corner, the "
            "hibiscus. Is that what you mean?’ The florist pointed at a leafy potted plant by the corner.\n\nThere, "
            "in a shaft of the wan afternoon sunlight, was a single blood-red blossom, its five petals splayed back to "
            "reveal a long stamen tipped with yellow pollen. Saeng felt a shock of recognition so intense, it was "
            "almost visceral. ‘Saebba,’ Saeng whispered.\n\nA saebba hedge, tall and lush, had surrounded their garden, "
            "its lush green leaves dotted with vermilion flowers. And sometimes after a monsoon rain, a blossom or two "
            "would have blown into the well, so that when she drew the well water, she would find a red blossom floating "
            "in the bucket.\n\nSlowly, Saeng walked down the narrow aisle toward the hibiscus. Orchids, lanna bushes, "
            "oleanders, elephant ear begonias, and bougainvillea vines surrounded her. Plants that she had not even "
            "realized she had known but had forgotten drew her back into her childhood world.\n\nWhen she got to the "
            "hibiscus, she reached out and touched a petal gently. It felt smooth and cool, with a hint of velvet "
            "toward the center—just as she had known it would feel.\n\nAnd beside it was yet another old friend, a "
            "small shrub with waxy leaves and dainty flowers with purplish petals and white centers. ‘Madagascar "
            "periwinkle,’ its tag announced. How strange to see it in a pot, Saeng thought. Back home it just grew wild, "
            "jutting out from the cracks in brick walls or between tiled roofs.\n\nAnd that rich, sweet scent—that was "
            "familiar, too. Saeng scanned the greenery around her and found a tall, gangly plant with exquisite little "
            "white blossoms on it. ‘Dok Malik,’ she said, savoring the feel of the word on her tongue, even as she "
            "silently noted the English name on its tag, ‘jasmine.’\n\nOne of the blossoms had fallen off, and "
            "carefully Saeng picked it up and smelled it. She closed her eyes and breathed in, deeply. The familiar "
            "fragrance filled her lungs, and Saeng could almost feel the light strands of her grandmother’s long gray "
            "hair, freshly washed, as she combed it out with the fine-toothed buffalo-horn comb. And when the sun had "
            "dried it, Saeng would help the gnarled old fingers knot the hair into a bun, then slip a dok Malik bud "
            "into it.\n\nSaeng looked at the white bud in her hand now, small and fragile. Gently, she closed her palm "
            "around it and held it tight. That, at least, she could hold on to. But where was the fine-toothed comb? "
            "The hibiscus hedge? The well? Her gentle grandmother?\n\nA wave of loss so deep and strong that it stung "
            "Saeng’s eyes now swept over her. A blink, a channel switch, a boat ride into the night, and it was all "
            "gone. Irretrievably, irrevocably gone.\n\nAnd in the warm moist shelter of the greenhouse, Saeng broke "
            "down and wept.\n\nIt was already dusk when Saeng reached home. The wind was blowing harder, tearing off the "
            "last remnants of green in the chicory weeds that were growing out of the cracks in the sidewalk. As if "
            "oblivious to the cold, her mother was still out in the vegetable garden, digging up the last of the onions "
            "with a rusty trowel. She did not see Saeng until the girl had quietly knelt down next to her.\n\nHer smile "
            "of welcome warmed Saeng. ‘Ghup ma laio le? You’re back?’ she said cheerfully. ‘Goodness, it’s past five. "
            "What took you so long? How did it go? Did you—?’ Then she noticed the potted plant that Saeng was holding, "
            "its leaves quivering in the wind.\n\nMrs. Panouvong uttered a small cry of surprise and delight. ‘Dok "
            "faeng-noi!’ she said. ‘Where did you get it?’ ‘I bought it,’ Saeng answered, dreading her mother’s next "
            "question. ‘How much?’ For answer Saeng handed her mother some coins.\n\n‘That’s all?’ Mrs. Panouvong "
            "said, appalled, ‘Oh, but I forgot! You and the Lambert boy ate Bee-Maags . . . .’ ‘No, we didn’t, Mother,’ "
            "Saeng said. ‘Then what else—?’ ‘Nothing else. I paid over nineteen dollars for it.’\n\n‘You what?’ Her "
            "mother stared at her incredulously. ‘But how could you? All the seeds for this vegetable garden didn’t "
            "cost that much! You know how much we—’ She paused, as she noticed the tearstains on her daughter’s cheeks "
            "and her puffy eyes. ‘What happened?’ she asked, more gently. ‘I—I failed the test,’ Saeng said.\n\nFor a "
            "long moment Mrs. Panouvong said nothing. Saeng did not dare look her mother in the eye. Instead, she stared "
            "at the hibiscus plant and nervously tore off a leaf, shredding it to bits.\n\nHer mother reached out and "
            "brushed the fragments of green off Saeng’s hands. ‘It’s a beautiful plant, this dok faeng-noi,’ she finally "
            "said. ‘I’m glad you got it.’ ‘It’s—it’s not a real one,’ Saeng mumbled.\n\n‘I mean, not like the kind we "
            "had at—at—’ She found that she was still too shaky to say the words at home, lest she burst into tears "
            "again. ‘Not like the kind we had before,’ she said.\n\n‘I know,’ her mother said quietly. ‘I’ve seen this "
            "kind blooming along the lake. Its flowers aren’t as pretty, but it’s strong enough to make it through the "
            "cold months here, this winter hibiscus. That’s what matters.’\n\nShe tipped the pot and deftly eased the "
            "ball of soil out, balancing the rest of the plant in her other hand. ‘Look how root-bound it is, poor "
            "thing,’ she said. ‘Let’s plant it, right now.’\n\nShe went over to the corner of the vegetable patch and "
            "started to dig a hole in the ground. The soil was cold and hard, and she had trouble thrusting the shovel "
            "into it. Wisps of her gray hair trailed out in the breeze, and her slight frown deepened the wrinkles around "
            "her eyes. There was a frail, wiry beauty to her that touched Saeng deeply.\n\n‘Here, let me help, Mother,’ "
            "she offered, getting up and taking the shovel away from her.\n\nMrs. Panouvong made no resistance. ‘I’ll "
            "bring in the hot peppers and bitter melons, then, and start dinner. How would you like an omelet with "
            "slices of the bitter melon?’ ‘I’d love it,’ Saeng said.\n\nLeft alone in the garden, Saeng dug out a hole "
            "and carefully lowered the ‘winter hibiscus’ into it. She could hear the sounds of cooking from the kitchen "
            "now, the beating of eggs against a bowl, the sizzle of hot oil in the pan. The pungent smell of bitter melon "
            "wafted out, and Saeng’s mouth watered. It was a cultivated taste, she had discovered—none of her classmates "
            "or friends, not even Mrs. Lambert, liked it—this sharp, bitter melon that left a golden aftertaste on the "
            "tongue. But she had grown up eating it and, she admitted to herself, much preferred it to a Big Mac.\n\nThe "
            "‘winter hibiscus’ was in the ground now, and Saeng tamped down the soil around it. Overhead, a flock of "
            "Canada geese flew by, their faint honks clear and—yes—familiar to Saeng now. Almost reluctantly, she "
            "realized that many of the things that she had thought of as strange before had become, through the quiet "
            "repetition of season upon season, almost familiar to her now. Like the geese. She lifted her head and "
            "watched as their distinctive V was etched against the evening sky, slowly fading into the distance.\n\nWhen "
            "they come back, Saeng vowed silently to herself, in the spring, when the snows melt and the geese return "
            "and this hibiscus is budding, then I will take that test again."
        ),
        "judge_rubric": (
            "Official 0–3 source-dependent-response rubric. Score 3 demonstrates understanding of textual "
            "complexities, addresses the question, uses expressed and implied information, and extends beyond the "
            "literal. Score 2 is partial/literal: it addresses the question and uses some evidence but may not connect "
            "support to a conclusion. Score 1 is minimal understanding, may misread text/question, and lacks supporting "
            "explanation. Score 0 is irrelevant, incorrect, or no response."
        ),
        "scored_construct": "source-dependent reading response",
        "raw_score_contract": "Each rater and resolved score is 0–3.",
    },
    7: {
        "source_sha256": "cbaa3e9fda7dad87061fc9b986cf9f087b0d5041cf8783b0301ccf1a05d35630",
        "source_file": "Essay%20Set%20%237--ReadMeFirst.docx",
        "task": (
            "Write about patience. Being patient means that you are understanding and tolerant. A patient person "
            "experiences difficulties without complaining. Do only one of the following: write a story about a time "
            "when you were patient; write a story about a time when someone you know was patient; or write a story in "
            "your own way about patience."
        ),
        "source_passage": "",
        "judge_rubric": (
            "Official trait rubric, 0–3 per trait. Ideas (doubled): 3 clearly focused and thoroughly developed with "
            "specific relevant details; 2 somewhat focused with mixed details; 1 minimal focus/limited general details; "
            "0 unfocused or undeveloped. Organization: clear/logically sequenced (3), logically sequenced (2), weak "
            "(1), or absent (0). Style: compelling varied language (3), adequate clear language (2), limited variety "
            "that may hinder purpose/audience (1), or ineffective (0). Conventions: consistent (3), adequate (2), "
            "limited (1), or ineffective (0) Standard English grammar, usage, spelling, capitalization and punctuation."
        ),
        "scored_construct": "narrative quality about patience",
        "raw_score_contract": "Per-rater composite is 0–12 (Ideas doubled); resolved score is 0–24.",
    },
    8: {
        "source_sha256": "c19006d15a97ea87b47e21e76f24813eeedc3c60c98b0c9c18e31fa1fae08ed1",
        "source_file": "Essay%20Set%20%238--ReadMeFirst.docx",
        "task": (
            "We all understand the benefits of laughter. For example, someone once said, ‘Laughter is the shortest "
            "distance between two people.’ Many other people believe that laughter is an important part of any "
            "relationship. Tell a true story in which laughter was one element or part."
        ),
        "source_passage": "",
        "judge_rubric": (
            "Official 1–6 trait rubric: Ideas/content must be clear, focused, interesting and supported by relevant "
            "details; organization must sequence ideas coherently with an effective beginning, transitions and closure; "
            "voice must fit purpose/audience and engage the reader; word choice must be precise and varied; sentence "
            "fluency must have effective rhythm and varied structure; conventions cover grammar, usage, spelling, "
            "capitalization and punctuation. Higher scores show stronger focus, development, organization, engagement, "
            "language control and conventions; lower scores are increasingly unclear, underdeveloped, disorganized, "
            "mechanical, or error-heavy. The official composite uses Ideas, Organization, Sentence Fluency, and "
            "Conventions (with Conventions double-weighted); Voice and Word Choice remain rubric context but are not "
            "in the reported composite formula."
        ),
        "scored_construct": "true narrative quality with laughter",
        "raw_score_contract": "Per-rater composite is 5–30; resolved score is 10–60.",
    },
}

# Source passages 3 and 4 are longer than the frozen 2,048-token Judge
# context once a student response is appended.  The complete archival source
# is identified above by its document SHA-256; these task-relevant source
# contexts are the bounded material that actually enters the template.
# This is intentionally explicit rather than allowing right-truncation to
# remove unknown task/rubric/essay tokens.
ASAP_PROMPT_CATALOG[3]["source_passage"] = (
    "Verified source context for ‘Rough Road Ahead’ by Joe Kurmaskie:\n\n"
    "The cyclist follows old men’s ‘shortcut’ advice from Lodi toward Yosemite. The first town is a ghost town; "
    "the sun is beating down. In the high deserts of California in June, he finds a ramshackle shed and rusty "
    "pumps. The water is brackish, extremely hot, and tastes like battery acid.\n\n"
    "The terrain changes from flat road to short rolling hills under a ROUGH ROAD AHEAD sign. At only 12 mph, the "
    "dehydrated cyclist meets tumbleweeds and a large snake; his water bottles have only a few sips left and he "
    "fears heatstroke. An abandoned grape-juice factory deepens his despair.\n\n"
    "After climbing away from the factory, he reaches Gary and Wilber’s Fish Camp, drinks deeply from a bathroom "
    "sink, and resolves that he will always use his own map rather than such travel advice."
)
ASAP_PROMPT_CATALOG[4]["source_passage"] = (
    "Verified source context for ‘Winter Hibiscus’ by Minfong Ho:\n\n"
    "Saeng and her family moved from Vietnam to the United States. After failing her driver’s test, Saeng sees a "
    "hibiscus in a florist shop. Its sight, smell, and familiar plants bring back memories of her family garden, "
    "grandmother, and home; she feels a wave of loss and weeps.\n\n"
    "At home, Saeng tells her mother she paid over nineteen dollars for the plant and failed the test. Her mother "
    "says the ‘winter hibiscus’ is not as pretty as the one at home but is strong enough to survive the cold months. "
    "Saeng plants it. While noticing that Canada geese have become familiar through repeated seasons, Saeng vows "
    "that in spring, when the geese return and the hibiscus buds, she will take the test again."
)


def prompt_catalog_metadata() -> dict[str, Any]:
    """Return the immutable catalog and template identities recorded in artifacts."""

    canonical = json.dumps(
        ASAP_PROMPT_CATALOG, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    catalog_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    template_payload = {
        "catalog_version": ASAP_PROMPT_CATALOG_VERSION,
        "template_version": ASAP_JUDGE_TEMPLATE_VERSION,
        "template_layout": _template_layout_contract(),
        "prompts": ASAP_PROMPT_CATALOG,
    }
    template_sha256 = hashlib.sha256(
        json.dumps(template_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "catalog_version": ASAP_PROMPT_CATALOG_VERSION,
        "catalog_sha256": catalog_sha256,
        "template_version": ASAP_JUDGE_TEMPLATE_VERSION,
        "template_sha256": template_sha256,
        "source_kind": "archival_competition_task_description_mirror",
        "source_base_url": _ARCHIVAL_SOURCE_BASE,
        "prompt_source_files": {
            str(prompt_id): {
                "url": _ARCHIVAL_SOURCE_BASE + str(entry["source_file"]),
                "sha256": str(entry["source_sha256"]),
            }
            for prompt_id, entry in sorted(ASAP_PROMPT_CATALOG.items())
        },
        "labels_in_prompt": False,
        "context_budget_policy": "full_task_requirement_compact_operational_rubric_bounded_source_context",
    }


def build_asap_judge_input(*, prompt_id: int, essay_text: str) -> str:
    """Build the frozen, label-free Judge input for one ASAP response."""

    entry = ASAP_PROMPT_CATALOG.get(int(prompt_id))
    if entry is None:
        raise ValueError(f"No frozen ASAP prompt/rubric entry for prompt {prompt_id}")
    metadata = prompt_catalog_metadata()
    source = str(entry["source_passage"]).strip()
    source_section = f"\n\nSOURCE EXCERPT:\n{source}" if source else ""
    return (
        "ASAP-AES overall-quality judging task\n"
        f"template_version: {metadata['template_version']}\n"
        f"template_sha256: {metadata['template_sha256']}\n"
        f"prompt_id: {int(prompt_id)}\n"
        f"scored_construct: {entry['scored_construct']}\n"
        f"raw_score_contract: {entry['raw_score_contract']}\n\n"
        f"TASK:\n{entry['task']}"
        f"{source_section}\n\n"
        f"SCORING RUBRIC:\n{entry['judge_rubric']}\n\n"
        "TARGET SCALE: Map the task's official raw score range to the common ordinal 1–5 scale by five equal-width "
        "buckets, with an exact boundary assigned to the higher bucket. Do not infer any score distribution from the "
        "essay collection.\n\n"
        "ESSAY TO SCORE:\n"
        f"{str(essay_text).strip()}\n\n"
        "OUTPUT CONTRACT: Produce only one integer from 1 through 5."
    )


def _template_layout_contract() -> list[str]:
    return [
        "ASAP-AES overall-quality judging task",
        "template_version",
        "template_sha256",
        "prompt_id",
        "scored_construct",
        "raw_score_contract",
        "TASK",
        "SOURCE EXCERPT when applicable",
        "SCORING RUBRIC",
        "TARGET SCALE",
        "ESSAY TO SCORE",
        "OUTPUT CONTRACT",
    ]
