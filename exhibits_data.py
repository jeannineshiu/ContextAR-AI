"""
Museum knowledge base — Western European Paintings, 15th–20th Century
All works sourced from the Metropolitan Museum of Art Open Access collection.

Each exhibit contains structured sections so the RAG engine can answer
questions from multiple angles (facts, visual, historical, technique, story).
"""

EXHIBITS = [
    {
        "id": "the_harvesters",
        "name": "The Harvesters",
        "artist": "Pieter Bruegel the Elder",
        "year": "1565",
        "type": "painting",
        "period": "Northern Renaissance, 1565",
        "met_accession": "19.164",
        "key_facts": (
            "Oil on wood panel, 116.5 × 159.5 cm. "
            "Painted 1565 by Pieter Bruegel the Elder. "
            "One of six panels depicting the seasons, commissioned by Antwerp merchant Niclaes Jonghelinck. "
            "Represents late summer (July–August). "
            "In the Metropolitan Museum of Art since 1919."
        ),
        "visual_description": (
            "A sweeping Flemish landscape under a warm, hazy summer sky fills the panel from edge to edge. "
            "In the foreground, a group of peasant workers rest beneath a large pear tree — one sleeps flat on the ground, "
            "others eat and drink from wooden bowls. To the left, more workers bend and swing scythes through a golden wheat field. "
            "The cut stubble in the middle ground catches the afternoon light. "
            "In the far distance, a village, a church spire, and a glimmering sea inlet dissolve into bluish haze. "
            "The palette is dominated by warm ochres, yellows, and earth greens, conveying the thick heat of a harvest day."
        ),
        "historical_context": (
            "The six-panel Months series was created for the private dining room of Niclaes Jonghelinck's estate near Antwerp. "
            "Each panel showed two months of the year. Only five of the six survive: The Harvesters (July–August), "
            "The Hunters in the Snow (January–February), The Gloomy Day (February–March), "
            "The Return of the Herd (October–November), and Haymaking (June–July). "
            "The series reflects Flemish humanist interest in the working calendar inherited from medieval Books of Hours, "
            "but Bruegel transformed the convention into monumental landscape painting. "
            "Jonghelinck later pledged the entire series as collateral on a debt; they were seized by the city of Antwerp in 1594."
        ),
        "technique": (
            "Bruegel worked in oil on oak panel — a common support in Northern Europe before canvas became dominant. "
            "His technique used thin, transparent glazes over opaque underpaint to create the luminous depth of sky and grain. "
            "The composition employs a high vantage point — what later theorists would call a 'bird's-eye view' — "
            "so the entire valley spreads before the viewer like a map. "
            "The figures in the foreground are painted with surprising solidity and individuality for the period; "
            "Bruegel gave peasant workers the same monumental presence as religious figures in earlier altarpiece traditions. "
            "This elevated vantage point and integration of figures into landscape was highly innovative in 1565."
        ),
        "story": (
            "Art historians were puzzled for centuries by the sleeping figure in the foreground — "
            "a man who appears to have simply dropped where he stood, overcome by heat and exhaustion. "
            "He is one of the most naturalistic figures in 16th-century painting, completely unposed and utterly human. "
            "Bruegel is thought to have sketched real Flemish farm workers directly from life, "
            "an unusual practice at a time when most artists worked from classical models. "
            "The panel was separated from the other Months series paintings centuries ago; "
            "it entered the Metropolitan Museum of Art in 1919 through the bequest of Henry G. Marquand."
        ),
        "content": (
            "The Harvesters is an oil-on-wood panel painting by Flemish Renaissance master Pieter Bruegel the Elder, "
            "created in 1565. Measuring 116.5 × 159.5 cm, it depicts late summer in a rolling Flemish landscape: "
            "peasant workers harvest wheat in the background while a group rests under a pear tree in the foreground. "
            "The painting belongs to a series of six panels representing the seasons, commissioned by Antwerp merchant "
            "Niclaes Jonghelinck. Bruegel's composition uses a dramatic high vantage point and a warm ochre palette "
            "to evoke the heat and rhythm of harvest time. The work is celebrated as a founding monument of Western "
            "landscape painting and has been in the Metropolitan Museum of Art since 1919."
        ),
    },
    {
        "id": "young_woman_water_pitcher",
        "name": "Young Woman with a Water Pitcher",
        "artist": "Johannes Vermeer",
        "year": "c. 1662",
        "type": "painting",
        "period": "Dutch Golden Age, c. 1662",
        "met_accession": "89.15.21",
        "key_facts": (
            "Oil on canvas, 45.7 × 40.6 cm. "
            "Painted around 1662 by Johannes Vermeer. "
            "One of only 34–36 paintings attributed to Vermeer in total. "
            "Entered the Metropolitan Museum of Art in 1889 — one of the first Vermeers in an American institution. "
            "Subject is likely a housemaid or mistress of the household."
        ),
        "visual_description": (
            "A young woman stands at a window, her left hand resting on a gleaming silver water pitcher "
            "and her right hand gently pushing open the leaded-glass window panel. "
            "She wears a white linen headdress and a blue jacket, typical of prosperous Dutch domestic dress. "
            "On the table beside her lie a blue cloth, a jewellery box, and a silver basin. "
            "A large wall map of the Seventeen Provinces of the Netherlands hangs on the back wall. "
            "Light enters from the upper-left window and falls softly across the woman's face, her headdress, "
            "and the surface of the pitcher, creating delicate gradations of shadow and luminosity."
        ),
        "historical_context": (
            "Vermeer worked exclusively in Delft, a prosperous trading city in the Dutch Republic. "
            "The Dutch Golden Age (roughly 1588–1672) saw extraordinary economic growth funded by global trade, "
            "and a new merchant class that commissioned domestic paintings — a genre almost unknown in Catholic Europe. "
            "The wall map in the painting was a fashionable status symbol: maps of the Dutch provinces declared "
            "civic pride and worldly awareness. "
            "The silver pitcher and basin were standard items in wealthy Dutch households, "
            "used for morning washing rituals. "
            "Vermeer left only about 36 paintings when he died in 1675 at age 43, deeply in debt."
        ),
        "technique": (
            "Vermeer's signature achievement was the rendering of diffused, cool natural light — "
            "the light that enters from a single window on the left side and wraps around every surface. "
            "He applied paint in extraordinarily fine layers, building from a warm reddish-brown ground "
            "up through translucent glazes to the bright highlights on the pitcher and headdress. "
            "The characteristic 'pointillé' effect — tiny dots of lighter paint on surfaces in light — "
            "suggests the sparkle of real materials. "
            "Scholars have long debated whether Vermeer used a camera obscura as a compositional tool; "
            "the perfectly judged perspective and the soft defocus on background elements are consistent with such a device. "
            "Whatever his method, no other painter of the period captured interior light with this precision."
        ),
        "story": (
            "When the painting was first sold to the Metropolitan Museum of Art in 1889, "
            "Vermeer was barely known outside the Netherlands. "
            "His rediscovery was largely the work of French critic Théophile Thoré, "
            "who in 1866 published a series of articles attributing dozens of unsigned Dutch paintings to 'Van der Meer of Delft.' "
            "Today Vermeer commands some of the highest prices at auction, "
            "but during his lifetime he appears to have struggled financially, "
            "running a modest art-dealing business alongside his painting. "
            "The identity of the woman in this and other Vermeer interiors remains unknown — "
            "she may have been a model hired from the neighbourhood, or a member of Vermeer's large household."
        ),
        "content": (
            "Young Woman with a Water Pitcher is an oil-on-canvas painting by Dutch Golden Age master Johannes Vermeer, "
            "created around 1662. A young woman stands at a window, her hand resting on a silver pitcher, "
            "opening the leaded-glass panel to admit soft daylight. Vermeer's genius lies in his exquisite rendering "
            "of diffused light falling across her linen headdress, blue jacket, and the gleaming metalware. "
            "A wall map of the Dutch provinces hangs behind her, signalling domestic prosperity. "
            "Of the 34–36 paintings attributed to Vermeer, this is among his finest small-scale interiors. "
            "It entered the Metropolitan Museum of Art in 1889, one of the first Vermeers in an American collection."
        ),
    },
    {
        "id": "aristotle_with_bust_of_homer",
        "name": "Aristotle with a Bust of Homer",
        "artist": "Rembrandt van Rijn",
        "year": "1653",
        "type": "painting",
        "period": "Dutch Golden Age, 1653",
        "met_accession": "61.198",
        "key_facts": (
            "Oil on canvas, 143.5 × 136.5 cm. "
            "Painted 1653 by Rembrandt van Rijn. "
            "Commissioned by Sicilian nobleman Don Antonio Ruffo, who simply requested 'a philosopher.' "
            "Acquired by the Metropolitan Museum of Art in 1961 for $2.3 million — "
            "at the time the highest price ever paid at auction for a painting."
        ),
        "visual_description": (
            "A richly dressed man — Aristotle — stands in three-quarter view against a deep, warm shadow. "
            "He wears a black costume with a gold chain draped across his chest, "
            "and a wide-brimmed black hat that casts his upper face in shade. "
            "His right hand rests gently on the top of a small white marble bust of a blind, elderly man — Homer. "
            "The gold chain bears a medallion portrait of Alexander the Great. "
            "Aristotle's face catches the warm light from the left, "
            "revealing an expression of deep, troubled contemplation. "
            "The bust of Homer, by contrast, is bathed in a cooler, almost spectral light, "
            "as if touched by a different, timeless radiance."
        ),
        "historical_context": (
            "Don Antonio Ruffo of Messina was one of the few patrons outside the Dutch Republic to commission works from Rembrandt. "
            "His 1652 letter requesting 'a philosopher' arrived when Rembrandt was at the height of his powers "
            "but already facing financial difficulties that would end in bankruptcy in 1656. "
            "Ruffo was so pleased with the result that he commissioned two more large canvases — "
            "an Alexander the Great and a Homer — creating a philosophical triptych. "
            "The subject reflects 17th-century humanist fascination with the tension between the active and contemplative life: "
            "Aristotle represents reason and earthly success (he tutored Alexander the Great); "
            "Homer represents pure creative imagination, blind to the world but immortal in verse."
        ),
        "technique": (
            "Rembrandt's late style is on full display: the background dissolves into almost formless shadow, "
            "while light falls with selective, dramatic intensity on the face, hands, and gold chain. "
            "The paint surface is physically rich — thick impasto ridges describe the chain's gold links and the fur trim, "
            "while thinly applied glazes build the shadow depths. "
            "X-ray analysis has revealed that Rembrandt significantly altered the composition during painting, "
            "including the position of the chain and the lighting of the figure. "
            "He often worked without preliminary drawings, composing directly on the canvas — "
            "a confident and unconventional approach even for his time."
        ),
        "story": (
            "When the Metropolitan Museum of Art paid $2.3 million for this painting in November 1961, "
            "the bid made front-page news around the world. "
            "The New York Times ran the headline 'Met Buys Rembrandt for Record $2.3 Million.' "
            "The purchase was controversial: critics questioned whether a public institution should pay "
            "such a sum for a single work, and the museum's director Thomas Hoving later called it "
            "'the most expensive painting in the world.' "
            "Today the painting anchors Room 964 of the Metropolitan Museum, "
            "where it draws more visitors than almost any other Dutch canvas. "
            "Ruffo's original inventory described it simply as 'a half-figure of a philosopher in the ancient manner' — "
            "the identification of the figure as Aristotle was made by art historians in the 20th century."
        ),
        "content": (
            "Aristotle with a Bust of Homer is an oil-on-canvas painting (143.5 × 136.5 cm) by Rembrandt van Rijn, "
            "completed in 1653. It depicts the Greek philosopher Aristotle resting his hand contemplatively "
            "on a marble bust of Homer, while wearing a gold chain bearing a medallion of Alexander the Great — "
            "his most famous pupil. The painting stages a meditation on knowledge, wealth, and creative genius: "
            "Aristotle, the man of reason and worldly success, seems to question whether Homer's timeless poetry "
            "represents a greater achievement. Rembrandt renders the scene with warm, theatrical light and a "
            "richly textured paint surface. The Metropolitan Museum acquired it in 1961 for $2.3 million, "
            "at the time the highest price ever paid at auction for a painting."
        ),
    },
    {
        "id": "madame_x",
        "name": "Madame X (Madame Pierre Gautreau)",
        "artist": "John Singer Sargent",
        "year": "1883–84",
        "type": "painting",
        "period": "Gilded Age Realism, 1883–84",
        "met_accession": "16.53",
        "key_facts": (
            "Oil on canvas, 208.6 × 109.9 cm. "
            "Painted 1883–84 by John Singer Sargent. "
            "Subject: Virginie Amélie Avegno Gautreau (1859–1915), a Louisiana-born Parisian socialite. "
            "Caused a public scandal at the Paris Salon of 1884. "
            "Sargent sold it directly to the Metropolitan Museum of Art in 1916, describing it as 'the best thing I have done.'"
        ),
        "visual_description": (
            "A tall, slender woman stands in near-profile against a warm brown background, "
            "her body turned slightly to the right with her chin tilted imperiously downward. "
            "She wears a form-fitting black evening gown with a deeply plunging neckline and jewelled shoulder straps. "
            "Her skin — powdered with lavender or pearl-white cosmetics — creates a stark tonal contrast "
            "against the dark dress and shadowed background. "
            "Her right hand grips the edge of a small table; her left arm hangs at her side. "
            "The composition is boldly simplified: no accessories, no elaborate setting, "
            "just figure, gown, skin, and darkness. "
            "The current painting shows the right shoulder strap in its upright position; "
            "in the original version exhibited in 1884, the strap hung provocatively off her shoulder."
        ),
        "historical_context": (
            "Virginie Gautreau was born in Louisiana and married a wealthy French banker, Pierre Gautreau. "
            "By the early 1880s she was famous in Paris for her unconventional beauty and her use of "
            "lavender-tinted face powder, which gave her skin an unusual, almost theatrical pallor. "
            "Sargent sought her out as a subject, convinced she would make a career-defining portrait. "
            "She agreed, and the two worked together through 1883 in a series of difficult sittings. "
            "When the portrait was exhibited at the Salon of 1884, the reaction was immediate and brutal: "
            "critics described it as 'a catastrophe' and 'a defiance of decorum.' "
            "Gautreau's mother visited Sargent's studio and begged him to withdraw the painting; he refused. "
            "The scandal effectively ended Sargent's career in Paris, and he relocated to London."
        ),
        "technique": (
            "Sargent was 27 when he began Madame X and already a technically brilliant painter trained in Paris. "
            "The portrait's power comes from radical simplification: most of the canvas is occupied by "
            "the flat, almost abstract expanse of black dress and brown background. "
            "The face and décolletage — the only large areas of light — attract the eye with irresistible force. "
            "Sargent used a thin, fluid paint application in the background and dress, "
            "contrasted with more loaded, almost sculptural brushwork on the face and neck. "
            "The small arc of the jewelled shoulder strap against pale skin is one of the most "
            "studied passages in 19th-century portraiture. "
            "The composition's stark linearity anticipates Art Deco and early modernist graphic design."
        ),
        "story": (
            "The most famous detail of Madame X is the shoulder strap. "
            "In Sargent's original version, the right strap had slipped — or been artfully posed — "
            "off Gautreau's shoulder, suggesting undress and shocking Salon visitors. "
            "Faced with the uproar, Sargent repainted the strap into its upright position, "
            "but the damage to both reputations was done. "
            "For decades the painting languished in Sargent's studio. "
            "When he finally agreed to sell it in 1916, he wrote to the museum: "
            "'I suppose it is the best thing I have done,' and suggested retitling it simply 'Madame X' "
            "to protect the subject's privacy — though in Paris society the identity of 'Madame X' "
            "had never been a secret. "
            "Gautreau died in 1915, one year before Sargent sold the portrait."
        ),
        "content": (
            "Madame X is a full-length portrait in oil on canvas (208.6 × 109.9 cm) by John Singer Sargent, "
            "painted 1883–84. It depicts Virginie Amélie Avegno Gautreau, a Louisiana-born Parisian socialite, "
            "in a black evening gown with jewelled straps, her lavender-powdered skin stark against the dark background. "
            "When exhibited at the Paris Salon of 1884, the portrait caused a scandal — critics were outraged by "
            "its provocative pose and an original version in which one shoulder strap hung off her shoulder. "
            "The scandal drove Sargent from Paris to London. He later described it as 'the best thing I have done,' "
            "and in 1916 sold it directly to the Metropolitan Museum of Art."
        ),
    },
    {
        "id": "wheat_field_cypresses",
        "name": "Wheat Field with Cypresses",
        "artist": "Vincent van Gogh",
        "year": "1889",
        "type": "painting",
        "period": "Post-Impressionism, 1889",
        "met_accession": "49.30",
        "key_facts": (
            "Oil on canvas, 73.2 × 93.4 cm. "
            "Painted June 1889 by Vincent van Gogh. "
            "Created while Van Gogh was a voluntary patient at the Saint-Paul-de-Mausole asylum, Saint-Rémy-de-Provence. "
            "Considered the finest of three versions Van Gogh painted of this composition. "
            "In the Metropolitan Museum of Art since 1949."
        ),
        "visual_description": (
            "A rolling Provençal wheat field fills the lower two-thirds of the canvas, "
            "its golden grain rippling in directional brushstrokes that suggest both wind and restless energy. "
            "At the left edge, a single cypress tree rises like a dark green flame — "
            "tall, tapered, and almost black against the sky. "
            "To the right, lower olive trees and distant lavender hills complete the panorama. "
            "Above everything, a turbulent sky fills with swirling white and blue-grey clouds, "
            "painted in the same spiral rhythm as the wheat below. "
            "The palette is intensely chromatic: acid yellow, deep viridian green, cobalt blue, and white — "
            "all applied in thick, directional impasto strokes that make the canvas surface physically dynamic."
        ),
        "historical_context": (
            "Van Gogh checked himself into the Saint-Paul-de-Mausole asylum in May 1889, "
            "shortly after the crisis in which he famously severed part of his own ear in Arles. "
            "During his twelve months at Saint-Rémy he was extraordinarily productive, completing over 150 paintings. "
            "He was allowed to work in a studio within the asylum grounds and was occasionally permitted "
            "to paint in the surrounding fields and olive groves. "
            "The cypress trees that dominated the landscape fascinated him: in a letter to his brother Theo "
            "he described them as 'beautiful as regards line and proportion, like an Egyptian obelisk... "
            "a splash of black in a sunny landscape.' "
            "He associated cypresses with death and eternity, a symbolism rooted in Mediterranean funerary tradition, "
            "yet in this painting they burn with an almost violent vitality."
        ),
        "technique": (
            "Van Gogh's mature technique — fully developed by the Saint-Rémy period — "
            "involves applying thick oil paint in short, curved, directional strokes "
            "that describe both the physical form of each object and its emotional energy. "
            "In this painting the strokes in the wheat follow its movement in the wind; "
            "the strokes in the sky spiral outward; the cypress is built from interlocking upward dashes. "
            "The canvas surface has a physical relief — paint stands up in ridges and peaks — "
            "that makes the work feel three-dimensional even in reproduction. "
            "Van Gogh worked rapidly, often completing a large canvas in a single session. "
            "He mixed colours on the canvas as much as on the palette, creating optically vibrant passages "
            "where complementary colours — yellow and violet, orange and blue — sit side by side."
        ),
        "story": (
            "Van Gogh painted three versions of this composition — one is in the National Gallery, London, "
            "one is in a private collection, and this is the Metropolitan's version, "
            "widely considered the finest. "
            "He wrote to Theo about the painting in late June 1889: "
            "'I have a canvas of cypresses with some ears of wheat, some poppies, a blue sky — "
            "it is as beautiful as last year's in colour contrasts.' "
            "Van Gogh died on 29 July 1890, just over a year after completing this painting. "
            "He sold only one painting in his lifetime. "
            "Today his works command hundreds of millions of dollars at auction — "
            "Wheat Field with Cypresses alone is estimated to be worth over $200 million."
        ),
        "content": (
            "Wheat Field with Cypresses is an oil-on-canvas painting (73.2 × 93.4 cm) by Vincent van Gogh, "
            "painted in June 1889 while he was a voluntary patient at the Saint-Paul-de-Mausole asylum near "
            "Saint-Rémy-de-Provence. A cypress tree rises like a dark flame at the left; a turbulent sky "
            "swirls above golden wheat — every element painted in Van Gogh's characteristic spiral brushstrokes. "
            "He wrote to his brother Theo that the cypresses were 'beautiful as regards line and proportion, "
            "like an Egyptian obelisk,' associating them with eternity. "
            "This is considered the finest of three versions he painted of the composition. "
            "Van Gogh died in July 1890, one year after completing it."
        ),
    },
    {
        "id": "the_card_players",
        "name": "The Card Players",
        "artist": "Paul Cézanne",
        "year": "c. 1890–95",
        "type": "painting",
        "period": "Post-Impressionism / Proto-Cubism, c. 1890–95",
        "met_accession": "61.101.1",
        "key_facts": (
            "Oil on canvas, 65.4 × 81.9 cm. "
            "Painted c. 1890–95 by Paul Cézanne. "
            "One of five versions Cézanne painted of card-playing peasants; others are in the Musée d'Orsay, "
            "the Courtauld Gallery, the Barnes Foundation, and a private collection. "
            "Directly inspired Picasso and Braque when they developed Cubism in 1907–09."
        ),
        "visual_description": (
            "Two men sit at a simple wooden table, each absorbed in a hand of cards. "
            "They face each other in near-perfect symmetry, divided by a wine bottle at the centre of the table. "
            "The man on the left wears a soft round hat and a blue jacket; the man on the right wears a stiff "
            "cylindrical hat and a tan jacket. "
            "Both figures are painted with heavy, blocky forms — their torsos are solid cylinders, "
            "their faces simplified into broad planes of colour. "
            "The background is sparse: a warm ochre wall, the edge of a curtain, "
            "and the faintest suggestion of a third figure or spectator at the left edge. "
            "There is no narrative drama — only the silent, concentrated gravity of the game."
        ),
        "historical_context": (
            "Cézanne painted the Card Players series between approximately 1890 and 1895, "
            "working at his family estate Jas de Bouffan near Aix-en-Provence. "
            "The models were local farmworkers employed on the estate — men Cézanne knew well. "
            "At the time, Cézanne was a reclusive figure largely ignored by the Parisian art world; "
            "his work had been rejected by the official Salon repeatedly, and he had retreated to Provence. "
            "The Card Players series was based on a 17th-century Flemish genre tradition — peasant card games "
            "were a common subject — but Cézanne stripped the scene of all anecdote and moralizing content, "
            "reducing it to pure formal relationships. "
            "The series was first seen publicly in 1895 at Ambroise Vollard's gallery in Paris, "
            "where they attracted the attention of a new generation of artists including Picasso."
        ),
        "technique": (
            "Cézanne's method was to build form through short, parallel brushstrokes of modulated colour — "
            "a technique he called 'modulation' as opposed to 'modelling.' "
            "Rather than shading from light to dark with a single colour, "
            "he used adjacent patches of slightly different hues — warm and cool, bright and muted — "
            "to create the illusion of three-dimensional form on a flat surface. "
            "The result gives his figures a sculptural, almost geometric solidity "
            "that photographic realism cannot match. "
            "In The Card Players, this method reduces the human body to its essential volumes: "
            "cylindrical hats, tubular arms, blocky torsos. "
            "Picasso later said that Cézanne 'was the father of us all' — "
            "meaning that Cubism's fragmentation of form grew directly from Cézanne's example."
        ),
        "story": (
            "The five versions of The Card Players form a fascinating progression: "
            "the larger early versions include three or more figures and a boy spectator; "
            "the later versions — including this one — strip the scene down to just two players, "
            "achieving maximum concentration and formal purity. "
            "In 2011, one version of The Card Players sold at Christie's for approximately $259 million "
            "to the Royal Family of Qatar — at the time the highest price ever paid for a work of art. "
            "Cézanne himself was largely unaware of his growing influence: "
            "he died in 1906, having worked in near-isolation for decades, "
            "just as the generation of Picasso and Matisse was beginning to study his work with reverence."
        ),
        "content": (
            "The Card Players is an oil-on-canvas painting (65.4 × 81.9 cm) by French Post-Impressionist "
            "Paul Cézanne, produced in the early 1890s. Two Provençal peasants sit at a table, each absorbed "
            "in a hand of cards, divided by a wine bottle at the centre. Cézanne renders them with the solid, "
            "geometric weight that is his hallmark — short directional brushstrokes model cylindrical hats, "
            "blocky torsos, and tubular arms. This method of reducing form to underlying geometry directly "
            "inspired Picasso and Braque when they developed Cubism in 1907–09. "
            "In 2011, another version of this series sold for approximately $259 million — "
            "at the time the highest price ever paid for a work of art."
        ),
    },
]
