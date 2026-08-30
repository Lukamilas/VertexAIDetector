import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from test_data import test_human, test_ai

# TRAINING DATA
# 0 = Human
# 1 = AI
# ==================================================

texts = [

    # HUMAN

    "Last Saturday I woke up early for a tennis tournament. I was nervous at first but eventually settled down and played well.",

    "Yesterday I studied for my math test for three hours. The first few problems were difficult but I started understanding them after some practice.",

    "My family went hiking over the weekend. It started raining halfway through the trail and we had to run back to the car.",

    "I spent most of the afternoon helping my neighbor move into a new apartment. It was tiring but satisfying.",

    "During tennis practice I kept hitting my backhand into the net. My coach helped me fix my timing.",

    "My dad started work early this morning and spent the day meeting with patients. He came home tired but happy with how things went.",

    "Last weekend I spent almost three hours trying to fix my bike. I thought the problem was the chain, but it turned out one of the gears was bent. After a lot of trial and error I finally got it working",
 
    "I was running late for school this morning because my alarm never went off  I ended up grabbing breakfast on the way out the door and barely made it to my first class before the bell rang.",

    "During tennis practice yesterday I kept missing my forehand long. My coach told me to focus on brushing up on the ball more, and after a few drills the shot started feeling much better.",

    "My family decided to try a new restaurant on Friday night. None of us knew what to order, so we ended up sharing several different dishes and comparing our favorites.",

    "I spent most of the afternoon helping my younger brother study for a science quiz. Explaining the material out loud actually helped me understand it better too.",

    "Last month I started keeping track of how much water I drink each day. I was surprised by how often I forgot to drink enough during school.",

    "A few days ago I found an old notebook from elementary school. Reading through it was funny because some of the things I wrote seemed incredibly important at the time.",

    "My friend and I played basketball at the park until it got dark. Neither of us kept score, but we stayed out there much longer than we planned.",

    "Yesterday I tried making pancakes for breakfast by myself. The first batch was overcooked, but the second one turned out much better.",

    "I stayed up later than usual finishing a project. By the time I was done I was tired, but it felt good knowing I wouldn't have to worry about it the next day.",

    "I chose this question because I am interested to find out how candidates use money to help gain an advantage over their opponents.",

    "My favorite depiction of the first scene of Macbeth was the 2010 version directed by Rupert Goold. I thought the scene was entertaining and also very creepy. The witches who are depicted as nurses in this version were different than in the other versions and it made the version of the play more attractive to watch.",

    "People still remake Shakespeare’s plays in modern times because he was such an influential figure in playwriting.",

    "The different plays all start with the three witches doing evil things and that lets the audience know that they are going to be the villains of the story.",

    "At the end of the story, Laurie’s mom starts to figure out that Laurie and Charles are the same person. At the beginning of the story she believes everything Laurie says and thinks that Charles is a real person that is doing all the bad things that Laurie says he is.",

    "In this dialogue, Laurie’s mom discovers that there is no Charles in the class and that Laurie made him up. He did this to tell his parents that he was doing all those bad things without them knowing it was him.",

    "In conclusion Laurie was rebelling against his over protective parents by misbehaving when he finally had freedom.",

    "In The Tragedy of Macbeth, the Weird Sisters used Macbeth's inner ambition to completely destroy his ability to think clearly and fundamentally eliminate his sense of integrity.",

    "His ambition took complete control over his thoughts and actions, fueled by the witches' unrealistic prophecies, leading to his overbearing credulity and predestined demise.",

    "Macbeth's undoing reveals the indubitable truth that ambition is a double-edged sword embedded deep in human nature; it can propel people much further than hard work on its own, but in the process, it pushes people to make extreme and unethical decisions that can be illicit or even tyrannical, destroying them from the inside of their own mind!",

    "One reason schools require community service is that it encourages students to become involved in their local communities and develop a sense of responsibility.",

    "Although technology has improved communication, it has also created new challenges related to privacy and the spread of misinformation.",

    "The main character changes significantly throughout the story because he learns from his mistakes and gains a better understanding of those around him.",

    "Many students believe homework should be limited because excessive assignments can reduce time available for extracurricular activities and family responsibilities.",

    "The author develops the theme of perseverance by showing how the protagonist continues working toward his goal despite repeated setbacks.",

    "Renewable energy sources have become increasingly important as governments seek ways to reduce dependence on fossil fuels.",

    "One advantage of public transportation is that it can reduce traffic congestion while also lowering environmental impacts.",

    "The evidence presented in the article supports the claim that regular exercise contributes to both physical and mental well-being.",

    "Throughout the novel, symbolism is used to represent the conflict between individual freedom and societal expectations.",

    "Historical events are often influenced by a combination of economic, political, and social factors rather than a single cause.",

    "The experiment demonstrated that changes in temperature can significantly affect the rate of a chemical reaction.",

    "Many people argue that social media has changed the way information is consumed and shared across society.",

    "The speaker effectively appeals to the audience by combining logical evidence with emotional examples.",

    "Education plays a critical role in preparing individuals for future careers and civic participation.",

    "While the policy may provide short-term benefits, its long-term consequences should also be carefully considered.",

    "The setting contributes to the mood of the story by creating a sense of isolation and uncertainty.",

    "Researchers continue to study the relationship between sleep habits and academic performance among students.",

    "The argument is strengthened by the use of credible sources and well-supported evidence.",

    "Economic growth can improve living standards, but it may also create challenges related to resource consumption.",

    "The events described in the passage illustrate the importance of cooperation when solving complex problems.",

    "Small businesses often face challenges when competing against larger corporations because they typically have fewer financial resources and less brand recognition.",

    "A company's reputation can have a significant impact on its ability to attract customers and maintain long-term success.",

    "Many entrepreneurs take substantial risks when starting a business because there is no guarantee that the venture will be profitable.",

    "Customer satisfaction is important because repeat customers are more likely to recommend a business to others.",

    "Effective leadership can improve employee morale and help an organization achieve its goals more efficiently.",

    "One reason businesses conduct market research is to better understand consumer preferences before launching a new product.",

    "The growth of online shopping has forced many traditional retailers to adapt their business strategies.",

    "Businesses that fail to innovate may struggle to remain competitive in rapidly changing industries.",

    "A clear mission statement can help guide decision-making and communicate a company's values to stakeholders.",

    "Strong communication between managers and employees often leads to a more productive work environment.",

    "Many companies invest in employee training programs because skilled workers can contribute more effectively to organizational success.",

    "Economic conditions can influence consumer spending habits and affect the profitability of businesses.",

    "Marketing campaigns are most effective when they target the specific needs and interests of potential customers.",

    "Business ethics play an important role in maintaining trust between organizations and the communities they serve.",

    "A company that consistently delivers high-quality products is more likely to develop customer loyalty over time.",

    "Globalization has created opportunities for businesses to expand into international markets and reach new customers.",

    "Successful businesses often balance short-term financial goals with long-term strategic planning.",

    "Managers must frequently make decisions based on limited information while considering potential risks and rewards.",

    "The use of technology has improved efficiency in many workplaces by automating repetitive tasks and simplifying communication.",

    "Corporate social responsibility initiatives can strengthen a company's public image while also benefiting society.",

    "The sun rose above the hills.",

    "A bird landed on the fence.",

    "Water freezes at zero degrees Celsius.",

    "The train arrived ten minutes late.",

    "She opened the window for fresh air.",

    "The cat slept on the sofa.",

    "Rain fell throughout the afternoon.",

    "The meeting starts at nine o'clock.",

    "Children played in the park.",

    "Coffee was served after dinner.",

    "The road curves around the lake.",

    "He forgot his keys at home.",

    "The store closes on Sundays.",

    "Snow covered the mountain peaks.",

    "The teacher wrote on the board.",

    "A candle flickered in the dark.",

    "The book was placed on the shelf.",

    "Spring flowers bloomed early this year.",

    "The dog chased a tennis ball.",

    "The package arrived this morning.",

    "The river flows toward the sea.",

    "Leaves drifted across the sidewalk.",

    "The phone rang twice.",

    "A truck passed by the house.",

    "The clock struck midnight.",

    "The garden needs more water.",

    "He sat near the fireplace.",

    "The bridge spans the valley.",

    "The sky turned orange at sunset.",

    "Birds migrate during the winter.",

    "The computer restarted automatically.",

    "The bus stopped at the corner.",

    "She carried a blue umbrella.",

    "The cake cooled on the counter.",

    "Wind rattled the old door.",

    "The lamp lit the room.",

    "Students gathered in the hallway.",

    "The path leads through the forest.",

    "A fish jumped from the water.",

    "The mail arrived before noon.",

    "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.",

    "We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills; we shall never surrender.",

    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.",

    "And so, my fellow Americans: ask not what your country can do for you—ask what you can do for your country.",

    "I have cherished the ideal of a democratic and free society in which all persons live together in harmony and with equal opportunities. It is an ideal which I hope to live for and to achieve.",

    "Look at me! Look at my arm! I have ploughed and planted, and gathered into barns, and no man could head me! And a'n't I a woman?",

    "Ours is not a drive for power, but purely a non-violent fight for India’s independence.",

    "Is life so dear, or peace so sweet, as to be purchased at the price of chains and slavery? Forbid it, Almighty God! I know not what course others may take; but as for me, give me liberty or give me death!",

    "The government of this country has to face this alternative; either they must kill women or they must give women the vote.",

    "I would say to the House, as I said to those who have joined this government: 'I have nothing to offer but blood, toil, tears and sweat.'",

    "The light has gone out of our lives and there is darkness everywhere.",

    "General Secretary Gorbachev, if you seek peace, if you seek prosperity for the Soviet Union and Eastern Europe, if you seek liberalization: Come here to this gate! Mr. Gorbachev, tear down this wall!",

    "Your time is limited, so don't waste it living someone else's life.",

    "Yesterday, December 7th, 1941—a date which will live in infamy—the United States of America was suddenly and deliberately attacked by naval and air forces of the Empire of Japan.",

    "You and I have a rendezvous with destiny. We'll preserve for our children this, the last best hope of man on earth, or we'll sentence them to take the last step into a thousand years of darkness.",

    "What, to the American slave, is your 4th of July? I answer; a day that reveals to him, more than all other days in the year, the gross injustice and cruelty to which he is the constant victim.",

    "You shall not press down upon the brow of labor this crown of thorns; you shall not crucify mankind upon a cross of gold.",

    "It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better. The credit belongs to the man who is actually in the arena...",

    "We do not need magic to transform our world, we carry all the power we need inside ourselves already: we have the power to imagine better.",

    "To those waiting with bated breath for that favorite media catchphrase, the 'U-turn', I have only one thing to say: 'You turn if you want to. The lady's not for turning.'",
    
    "In *Fences* by August Wilson, talent versus hard work is a common theme throughout the book. This theme is used to show how the main character, Troy, faced challenges early in life and how those challenges affect him.",

    "In Act one, scene three, Troy's son, Cory, reveals his dreams to play college football; however, Troy is haunted by his inability to make a living in baseball due to racism. Troy says, \" '[t]he white man ain't gonna let you get nowhere with that football noway' \" (Wilson 35), revealing his value of hard work rather than talent.",

    "The phrase \"ain't gonna let you\" shows how making a living goes deeper than talent, while the phrase \"nowhere\" suggests complete restriction. Also, \"the white man\" not letting Cory play football is indicative of Troy's fear of racism.",

    "Instead of letting Cory play football and follow his dreams, Troy's desire for him is to find a solid job that pays well. In conclusion, the theme of hard work versus talent shapes how Troy lives his life.",

    "In 2026 gender roles are much less defined; however, during Shakespeare's era, they were more specific. This is demonstrated by the characters in Shakespeare's play *Macbeth.*",

    "Although Lady Macbeth is expected to follow the female gender roles, she rejects these stereotypes. On the other hand, Macbeth conforms to the expectations of a man during this time period.",

    "In Shakespeare’s *Macbeth,* Elizabethan female and male gender roles can be seen through the words and actions of Lady Macbeth and Macbeth. Throughout the play, Lady Macbeth defies women’s gender roles.",

    "During earlier parts of the play, the reader sees Lady Macbeth step into a male role when planning to murder Duncan with Macbeth. When Macbeth hesitates and decides murdering Duncan is the wrong decision, Lady Macbeth says, “When you durst do it, then you were a man; / And to be more than what you were, you would/ Be so much more the man. Nor time nor place” (1.7.56–58).",

    "Lady Macbeth demonstrates her “unwomanly” personality by threatening Macbeth’s manliness. Lady Macbeth manipulates Macbeth into being threatened by her decisiveness and strong statements.",

    "This switches her into playing the role of the man in their relationship. Lady Macbeth doesn’t accept the conventional female role in a marriage.",

    "Instead she acts like a man telling Macbeth, “be so much more than a man,” Lady Macbeth then informs Macbeth that he needs to be more of a man. Women in the Elizabethan era were expected to be subservient.",

    "In addition, women were thought to need someone to take care of them and look after them (Shanette). Clearly, Lady Macbeth goes against this stereotype, and instead, tells her husband what he needs to do.",

    "She stands up to Macbeth showing the reader she isn't scared to challenge his authority. Finally, Lady Macbeth doesn’t do what she is expected to: defying women's gender roles.",

    "During the play Macbeth alternates between rejecting male gender roles and doing what is expected of a man. Towards the beginning of the book Macbeth questions his manhood by going back on his plan to kill Duncan.",

    "He is shown being “weak” which doesn’t represent how a man should've acted back in the Elizabethan era. Back in the Elizabethan era men were privileged over women and upper class(rich) men were thought to be above other lower class men.",

    "Men were supposed to be the head of the house and do three things to enter manhood; control the discipline in their family, show authority, and control women (Shepard). Macbeth, in the beginning of the play, does the complete opposite letting Lady Macbeth discipline him and threaten him.",

    "By letting Lady Macbeth do this, Macbeth fails to complete the male role in the house. Macbeth rejects his role as a man by not showing authority.",

    "This reveals to the reader that Macbeth doesn’t conform to the expectations of how a man should act. Later in the play, he comes to realize that he has lost control of his masculinity and he decides to start acting like a “man”.",

    "Lady Macbeth challenges Macbeth's decisions, and Macbeth shows his new masculinity by saying “Do not muse at me, my most worthy friends./I have a strange infirmity, which is nothing/To those that know me…Then I’ll sit down.-Give me some wine. Fill full”(3.4.-103-105, 107). Macbeth doesn’t take any criticism from Lady Macbeth, keeping her from controlling him.",

    "Also, he commands her to bring him a drink which would be a very typical thing for a man to do in his time: making the woman of the house do most of the work while he sits and relaxes. In conclusion, Macbeth goes from not meeting male expectations to playing the role of a a-typical man.",

    "Both Macbeth and Lady Macbeth defy gender stereotypes at some point in play. At one point, Macbeth defies gender roles when he allows Lady Macbeth to control him and tell him what to do.",

    "In addition, Lady Macbeth doesn't wait on Macbeth and states her mind showing she isn't scared to express her thoughts to him. Unfortunately, Gender roles are meant to undermine people and don’t allow them the freedom they deserve.",

    "Something that I care about beyond myself is tennis because it is my favorite sport. I am also excited to learn; however, I find some injustices in the school system, for example favoring students and many other things.",

    "This and other injustices make me want to take action. In regards to school, I am especially excited about this class and the diploma in entrepreneurial studies.",

    "I am glad that it started this year; the year that I finished freshman year. One thing I want to protect is integrity among students and sports.",

    "For example, there are cheaters in tennis as there are in school. I find this unfair, and it makes me want to take action.",

    "If we could preserve integrity, I think that it would be very helpful for both the sports and the school system as a whole. Lastly, I also think that protecting nature is important.",

    "The story of Macbeth is more similar to the world today than you may think. This is because of: unchecked power, arrogant leaders, and countries in chaos.",

    "Throughout the story we see Macbeth rise to power through dark ways; moreover, things such as murdering king duncan and executing so-called “traitors”. This rise to power quickly vanishes due to his arrogance and his idea of being superior to everyone and everything, even the supernatural.",

    "Macbeth leaves his castle to get clarification from the witches on the prophecy they gave him. The witches tell Macbeth that he will die when the forest reaches the castle.",

    "Macbeth, being the arrogant man he is, thinks he will never die, claiming: \"That will never be/... high-placed Macbeth/Shall live the lease of nature, pay his breath/ To time\" (IV.i.108,112-14). Macbeth is extremely arrogant and feels untouchable.",

    "Being told he cannot die until Burnam Wood reaches the castle he feels like he \"shall live the lease of nature\". Macbeth uses the word \"highplaced\" because he feels better than everyone.",

    "Macbeth only cares about himself and the good news the witches give him, never looking at the potentially bad side of it. However, this all-powerful feeling Macbeth has is quickly washed away by the trueness of the witches' warnings.",

    "Macbeth sitting in his castle room is interrupted by a servant bearing bad news. The servant tells Macbeth there is an army outside the castle walls disguising themselves with tree branches.",

    "Macbeth, believing it's impossible, tells the servant to \"Go prick thy face... [Macbeth realises the severeness] I have almost forgotten the taste of fears\" (vi.vi.11). Macbeth refuses to trust anyone, he tells someone who could possibly save his life to “Go prick thy face”, effectively proving himself as an evil dictator.",

    "Once realization hits a wave of fear washes over him, this is proven through the fact that he said: “I have almost forgotten the taste of fear”; moreover, his use of almost shows that there is still a small amount of humanity left in him. Macbeth's arrogance is what caused his death because he did not listen to the warnings.",

    "Overall, arrogance is what proved to be what killed Macbeth. Whether it was not listening to: his own servants, supernatural witches, or even himself, in the end it was his failure to rely on others because he believed he was superior.",

    "Not only did he rise through completely immoral ways, he also became completely unfazed by all kinds of violence; furthermore, that is why it’s so shocking that he was so fearful on the day of his death. In the vignette “Laughter”, the author, Sandra Cisneros, uses similes and metaphors to show how the main character, Esperanza, relates to her sister, Nenny.",

    "Shortly before Esperanza goes to Gil’s Furniture Store, she has a flashback to driving by a house with her family. Esperanza says, “[l]ook at that house,…it looks like Mexico.”",

    "All the other siblings laugh, but Nenny replies: “That’s Mexico all right. That’s what I was thinking exactly” (Cisneros 18). In these quotes, Esperanza uses the simile “looks like Mexico”.",

    "Two of Esperanza’s siblings laugh, but Nenny relates to Esperanza by using the metaphor, “[t]hat’s Mexico all right”. This vignette is important because it establishes an important connection between Esperanza and Nenny.",
    
    #AI

    "Artificial intelligence is becoming increasingly important across a wide range of industries. Organizations use advanced technologies to improve efficiency and productivity.",

    "Furthermore, businesses can leverage technology to optimize operations and gain valuable insights from large datasets.",

    "Technological innovation plays a crucial role in long-term economic growth and organizational success.",

    "Organizations must adapt to rapidly evolving market conditions in order to remain competitive and sustainable.",

    "Data-driven decision making enables companies to identify opportunities and address emerging challenges effectively.",

    "Modern digital solutions provide businesses with the tools necessary to streamline processes and enhance performance.",

    "Artificial intelligence continues to transform industries by enabling organizations to automate processes, improve efficiency, and enhance decision-making capabilities.",

    "Effective communication plays a crucial role in organizational success. By fostering collaboration and encouraging transparency, teams can achieve their objectives more efficiently.",

    "Technological innovation has significantly impacted modern society by creating new opportunities for economic growth and productivity.",

    "Furthermore, businesses can leverage data-driven strategies to optimize performance and identify emerging market opportunities.",

    "In today's rapidly evolving digital landscape, organizations must remain adaptable in order to maintain a competitive advantage.",

    "Sustainable development requires a balanced approach that considers environmental, economic, and social factors simultaneously.",

    "Data analytics enables companies to gain valuable insights into consumer behavior and make more informed strategic decisions.",

    "Moreover, advancements in machine learning continue to drive innovation across a variety of sectors and applications.",

    "Organizations that embrace modern technologies are often better positioned to improve operational efficiency and achieve long-term success.",

    "The implementation of advanced technological solutions can help businesses streamline workflows and enhance overall productivity.",
 
    "The sun rose gently over the horizon, painting the sky with shades of orange and pink. Birds chirped as a new day began.",

    "Reading books can expand your knowledge and imagination. It is a simple habit that offers lifelong benefits.",

    "Exercise helps improve both physical and mental health. Even a short daily walk can make a difference.",

    "Technology has transformed the way people communicate. Messages can now travel across the world in seconds.",
        
    "Trees play an important role in the environment. They provide oxygen, reduce pollution, and support wildlife.",
        
    "Learning a new skill requires patience and practice. Consistent effort often leads to steady improvement.",
        
    "Good teamwork allows people to achieve goals more effectively. Collaboration encourages creativity and problem-solving.",
 
    "Healthy eating provides the body with essential nutrients. A balanced diet can help maintain energy throughout the day.",

    "Traveling introduces people to different cultures and perspectives. It can be both educational and enjoyable.",

    "Kindness is a small act that can have a big impact. Simple gestures of respect and compassion can brighten someone's day.",

    "Organizations can improve operational efficiency by implementing data-driven processes and continuously monitoring performance metrics.",

    "Artificial intelligence is increasingly being integrated into business operations to automate repetitive tasks and enhance productivity.",

    "Technological advancements have created new opportunities for innovation across a wide range of industries.",

    "Effective communication is essential for maintaining collaboration and ensuring that organizational objectives are achieved.",

    "The adoption of digital technologies has transformed how companies interact with customers and manage resources.",

    "Sustainable practices can help organizations reduce costs while also supporting long-term environmental goals.",

    "Data analytics enables decision-makers to gain valuable insights and improve overall performance.",

    "Educational institutions continue to explore new methods for integrating technology into the learning process.",

    "Modern transportation systems play a critical role in supporting economic development and global trade.",

    "The use of automation can increase consistency and reduce the likelihood of human error in many processes.",

    "Organizations that prioritize innovation are often better positioned to adapt to changing market conditions.",

    "Machine learning algorithms can identify patterns within large datasets that may not be immediately apparent to humans.",

    "Strategic planning allows businesses to allocate resources effectively and pursue long-term objectives.",

    "The growth of renewable energy technologies has contributed to significant changes in the global energy sector.",

    "Customer feedback provides valuable information that can guide product development and service improvements.",

    "Healthcare providers increasingly utilize digital systems to improve patient outcomes and streamline operations.",

    "Economic growth is influenced by a combination of technological innovation, investment, and consumer demand.",

    "Organizations frequently evaluate performance indicators to measure progress toward established goals.",

    "Cloud computing offers scalable solutions that support flexibility and operational efficiency.",

    "The development of new technologies often creates opportunities for both businesses and consumers.",

    "Research suggests that collaboration can improve problem-solving and encourage creative thinking.",

    "Digital communication platforms have significantly changed the way information is shared and accessed.",

    "Businesses must carefully balance risk and reward when making strategic decisions.",

    "Innovation frequently occurs when organizations identify unmet needs and develop effective solutions.",

    "The availability of information has expanded dramatically as internet access has become more widespread.",

    "Many industries have adopted advanced software tools to improve productivity and reduce costs.",

    "Organizations that invest in employee development may benefit from improved performance and retention.",

    "Data security remains a critical concern as digital systems become increasingly interconnected.",

    "Artificial intelligence applications continue to expand across healthcare, education, and finance.",

    "The effectiveness of a policy often depends on how well it is implemented and evaluated.",

    "Businesses frequently use performance metrics to assess progress and identify areas for improvement.",

    "Technological innovation can contribute to economic growth by increasing efficiency and creating new markets.",

    "Effective leadership involves establishing clear goals and supporting employees in achieving them.",

    "Global trade has increased the exchange of goods, services, and information between countries.",

    "Organizations often seek competitive advantages through innovation, efficiency, and customer satisfaction.",

    "The integration of digital tools has transformed many aspects of daily life and professional work.",

    "Long-term success often requires organizations to remain adaptable and responsive to changing conditions.",

    "Data-driven decision making allows leaders to evaluate evidence before selecting a course of action.",

    "Advances in technology continue to shape the future of communication, business, and education.",

    "Tiny drones mapped the orchard before sunrise.",

    "A forgotten lighthouse blinked across the foggy channel.",

    "The algorithm sorted memories like scattered photographs.",

    "Wildflowers pushed through cracks in the pavement.",

    "An electric scooter hummed past the mural.",

    "The glacier reflected shades of silver and blue.",

    "Three satellites crossed the night sky in formation.",

    "The baker experimented with rosemary and lemon.",

    "Dust swirled behind the off-road vehicle.",

    "A violin echoed through the empty station.",

    "The robot paused to recharge its battery.",

    "Moonlight traced patterns across the courtyard stones.",

    "A scientist recorded data during the storm.",

    "The canoe drifted beside a field of reeds.",

    "Fresh paint brightened the narrow hallway.",

    "The telescope revealed a distant spiral galaxy.",

    "An old map surfaced in the attic.",

    "The chef balanced sweet and smoky flavors.",

    "Solar panels lined the warehouse roof.",

    "A fox watched quietly from the hillside.",

    "The startup tested prototypes late into the evening.",

    "Cloud shadows moved across the desert floor.",

    "A musician tuned her guitar backstage.",

    "The orchard produced a record harvest.",

    "Neon signs reflected in the rain-soaked street.",

    "The submarine descended into colder waters.",

    "A cyclist followed the winding coastal road.",

    "The museum unveiled a restored sculpture.",

    "Several hikers crossed the suspension bridge.",

    "The spacecraft adjusted its trajectory automatically.",

    "Tall grass swayed beside the railway tracks.",

    "A programmer fixed the bug before release.",

    "The waterfall thundered through the canyon.",

    "An artist layered colors to create depth.",

    "The drone captured images of the shoreline.",

    "A comet appeared briefly before dawn.",

    "The research team compared multiple simulations.",

    "Lanterns glowed along the festival route.",

    "The pianist improvised a gentle melody.",

    "A storm cell formed beyond the horizon.",
    
    "The old, neon-blue sign outside the diner had been buzzing in a rhythmic, hypnotic broken beat for as long as anyone could remember.",

    "Inside, the scent of fresh coffee and maple syrup hung heavy in the air, wrapping around the vinyl booths like a warm blanket.",

    "A ceiling fan spun lazily overhead, slicing through the warm afternoon light that poured in through the dusty blinds.",

    "Outside, the world was rushing by in a blur of traffic and ringing phones, but in here, time seemed to stretch out, offering a brief, quiet pause from the noise.",
  
    "Artificial intelligence systems can process large amounts of information in a fraction of the time required by traditional methods.",

    "The rapid expansion of digital technology has changed how people communicate, work, and access information.",

    "Organizations frequently use performance indicators to evaluate progress and improve decision-making processes.",

    "Renewable energy sources are becoming increasingly important as countries seek sustainable alternatives to fossil fuels.",

    "The effectiveness of a policy depends on both its design and its implementation.",

    "Modern transportation networks enable goods and services to move efficiently across long distances.",

    "Data analysis can reveal patterns that help businesses better understand customer behavior.",

    "Educational institutions continue to explore innovative teaching methods that support student engagement.",

    "Technological advancements have created opportunities that were previously considered impossible.",

    "Effective leadership requires communication, adaptability, and strategic thinking.",

    "Machine learning models improve their performance by identifying relationships within data.",

    "Economic growth is often influenced by investment, productivity, and consumer demand.",

    "The internet has dramatically increased access to information for people around the world.",

    "Organizations must adapt to changing conditions in order to remain competitive.",

    "Scientific research contributes to a deeper understanding of complex natural phenomena.",

    "Digital platforms provide tools that facilitate collaboration among individuals and organizations.",

    "The development of new technologies often creates both opportunities and challenges.",

    "Many industries have adopted automation to increase efficiency and reduce costs.",

    "Environmental sustainability has become a major consideration in long-term planning efforts.",

    "The implementation of effective policies can improve outcomes across multiple sectors.",

    "Cloud computing enables organizations to store and access information remotely.",

    "Consumer preferences frequently evolve in response to cultural, economic, and technological changes.",

    "Businesses often rely on forecasting techniques when planning future operations.",

    "Innovation can occur when existing ideas are combined in new and creative ways.",

    "Public infrastructure plays a critical role in supporting economic activity.",

    "Digital communication tools allow information to be shared almost instantly.",

    "Organizations benefit from establishing clear goals and measurable objectives.",

    "Advances in healthcare technology have improved diagnostic and treatment capabilities.",

    "The availability of reliable data supports more informed decision making.",

    "Artificial intelligence applications continue to expand across numerous industries.",

    "A comprehensive strategy often includes both short-term and long-term objectives.",

    "Global markets are interconnected through trade, investment, and communication networks.",

    "The success of a project depends on planning, execution, and evaluation.",

    "Technological innovation can significantly increase productivity and efficiency.",

    "Data security remains a priority for organizations that manage sensitive information.",

    "The adoption of digital tools has transformed many traditional business practices.",

    "Research findings can help guide future policy and investment decisions.",

    "Organizations frequently seek methods to improve operational performance.",

    "The use of advanced analytics supports evidence-based decision making.",

    "Automation technologies are capable of performing repetitive tasks with consistency.",

    "Educational outcomes can be influenced by a variety of environmental and social factors.",

    "The global economy is shaped by interactions among governments, businesses, and consumers.",

    "Strategic planning enables organizations to allocate resources effectively.",

    "Technological progress often drives changes in workforce requirements.",

    "Many modern systems rely on interconnected networks to function efficiently.",

    "Continuous improvement initiatives focus on identifying opportunities for optimization.",

    "The integration of artificial intelligence has accelerated innovation in several fields.",

    "Organizations that embrace change are often better prepared for future challenges.",

    "Digital transformation has altered how information is collected, processed, and distributed.",

    "Emerging technologies continue to influence economic, social, and industrial development.",

    "To successfully navigate the rapidly evolving landscape of the modern global economy, future-ready enterprises must seamlessly operationalize cutting-edge, cloud-native architectures to maximize efficiency and foster high-impact collaboration.",

    "By harnesssing the power of actionable, granular insights and executing agile pivot strategies, organizations can streamline overhead costs while simultaneously elevating the end-to-end customer experience.",

    "Ultimately, nurturing a culture of continuous optimization and technological synthesis is what empowers market leaders to scale effectively, mitigate systemic risk, and redefine industry benchmarks.",

    "I think one of the biggest advantages of technology is that it allows people to communicate and share information much more easily.",

    "The weather was pretty nice today, and I decided to take a walk because I wanted to get some fresh air and clear my mind.",

    "Artificial intelligence has become an increasingly important part of modern society, influencing education, communication, business, and many other areas.",

    "I went through my notes a couple of times before the test, which helped me feel more prepared and confident about the material.",

    "One important factor to consider is how technology can be used responsibly while still allowing people to benefit from its many advantages.",

    "The movie was entertaining, although some parts felt predictable and the ending could have been developed in greater detail.",

    "When students have access to effective resources, they can often develop a deeper understanding of difficult concepts and become more independent learners.",

    "I wasn't completely sure what to do at first, but after looking at the instructions again, the assignment became much easier to understand.",

    "Social media has changed communication by allowing people to share ideas instantly and connect with others regardless of physical distance.",

    "The team had several opportunities to score, but better communication and more consistent execution would have improved the overall performance.",

    "I think the main reason the project worked was because everyone had a clear responsibility and understood what they needed to accomplish.",

    "Digital tools can make everyday tasks more efficient by reducing the amount of time required to organize information and complete routine activities.",

    "After reviewing the situation, it became clear that the original plan needed to be adjusted in order to achieve a more effective outcome.",

    "The book provides an interesting perspective on the importance of relationships and demonstrates how individual decisions can influence larger events.",

    "I decided to finish my homework before playing games because completing my responsibilities first allowed me to relax afterward without worrying about schoolwork.",

    "In today's increasingly connected world, individuals have greater access to information than ever before, creating both opportunities and challenges.",

    "The experiment produced results that supported the original hypothesis and demonstrated the relationship between the variables being studied.",

    "Although the assignment seemed difficult initially, breaking it into smaller sections made the overall process much more manageable.",

    "Businesses that effectively respond to changing consumer expectations are more likely to maintain long-term success in competitive markets.",

    "I usually prefer working in the morning because I can concentrate more easily when I have fewer distractions around me.",

    "Education plays an essential role in preparing individuals for future opportunities by developing knowledge, critical thinking, and communication skills.",

    "The situation could have been handled differently if the people involved had communicated their concerns earlier and worked toward a shared solution.",

    "I looked over the problem several times before realizing that I had misunderstood one of the instructions.",

    "Environmental sustainability requires long-term cooperation between governments, organizations, and individuals in order to produce meaningful results.",

    "The presentation was successful because the information was organized clearly and the main points were supported with relevant examples.",

    "Technology can provide valuable opportunities for students, particularly when digital resources are used to supplement rather than replace effective instruction.",

    "I wasn't expecting the assignment to take so long, but once I started working through it, I realized there were more steps than I originally thought.",

    "Effective leadership involves more than simply making decisions; it also requires communication, accountability, and the ability to respond to changing circumstances.",

    "The results suggest that small changes in behavior can sometimes produce meaningful differences over an extended period of time.",

    "I tried a different approach after the first one didn't work, and eventually I was able to figure out what the problem was.",

    "Modern organizations increasingly rely on data to identify trends, evaluate performance, and make decisions based on measurable evidence.",

    "The character's decision was important because it changed the direction of the story and affected the relationships between several other characters.",

    "I think having a schedule makes it easier to keep track of everything, especially when several assignments are due around the same time.",

    "Scientific research allows society to develop a better understanding of complex issues while providing evidence that can guide future decisions.",

    "The most effective solution would likely involve addressing the underlying cause of the problem rather than simply responding to its immediate effects.",

    "I finished the first part quickly, but the second section took much longer because I had to think more carefully about my answer.",

    "Communication is an important component of successful teamwork because it allows individuals to coordinate their efforts and resolve potential misunderstandings.",

    "The increasing availability of online resources has made it easier for people to access educational information outside traditional classroom environments.",

    "I wasn't sure whether my answer was correct, so I went back through the steps and checked each part of my work.",

    "Organizations can improve their performance by identifying areas that need improvement and developing strategies that address those weaknesses.",

    "The author's use of specific examples helps communicate the central argument and makes the overall message easier for readers to understand.",

    "I decided to take a break after studying for a while because I noticed that I was having trouble concentrating on the material.",

    "Artificial intelligence can support productivity by automating repetitive tasks and helping individuals analyze large amounts of information more efficiently.",

    "The decision ultimately produced positive results, although several unexpected challenges had to be addressed along the way.",

    "I thought the test would be harder than it was, but reviewing the material beforehand definitely made me feel more prepared.",

    "A balanced approach is often necessary when addressing complex problems because different perspectives can reveal potential solutions that might otherwise be overlooked.",

    "The development of new technologies continues to influence the workplace by changing how people communicate, collaborate, and complete tasks.",

    "I started with the easiest questions and then returned to the harder ones once I had completed everything I felt confident about.",

    "The evidence supports the conclusion that environmental conditions can have a significant influence on the behavior and development of living organisms.",

    "Successful projects generally require clear objectives, effective planning, and consistent communication among everyone involved.",

    "I didn't really understand the topic at first, but seeing a few examples helped me understand how the different ideas were connected.",

    "Businesses can use customer feedback to identify areas for improvement and develop products or services that better address consumer needs.",

    "The story demonstrates how seemingly small decisions can eventually lead to significant consequences for the people involved.",

    "I usually write down the things I need to finish because otherwise I sometimes forget about smaller assignments until the last minute.",

    "The responsible implementation of artificial intelligence requires consideration of accuracy, privacy, transparency, and potential effects on individuals.",

    "The results of the study indicate that additional research may be necessary before a definitive conclusion can be reached.",

    "I changed my original answer after looking at the problem again and realizing that I had overlooked an important detail.",

    "Technology has created new opportunities for collaboration by allowing people to work together even when they are located in different places.",

    "The central argument becomes more convincing when it is supported by specific evidence rather than relying solely on general statements.",

    "I didn't expect the practice session to be so helpful, but repeating the same skill several times made it feel much more natural.",

    "Economic conditions can influence consumer behavior by affecting purchasing decisions, confidence, and the amount of money people are willing to spend.",

    "The proposed solution addresses several of the major challenges while also creating opportunities for future improvement.",

    "I had trouble getting started because I wasn't sure how to organize my ideas, but making a quick outline helped.",

    "Students can benefit from personalized learning strategies because different individuals may require different approaches to understand the same material.",

    "The final result was different from what I originally expected, which provided an opportunity to reconsider the assumptions behind my prediction.",

    "I went back and checked my work before submitting it because I wanted to make sure I hadn't made any simple mistakes.",

    "The continued development of digital technology is likely to influence education by changing how students access information and interact with learning materials.",

    "The author's perspective becomes clearer throughout the passage as additional evidence is introduced to support the main idea.",

    "I prefer studying somewhere quiet because background noise usually makes it harder for me to stay focused.",

    "Organizations that prioritize innovation may be better positioned to adapt to changes in technology, consumer behavior, and competitive conditions.",

    "The experiment demonstrated that changing one variable can produce measurable effects on another variable under controlled conditions.",

    "I thought I had finished everything, but then I remembered that there was one smaller assignment I still needed to complete.",

    "Effective problem solving often requires individuals to evaluate multiple possibilities before selecting the approach that is most appropriate.",

    "The use of digital communication has made it possible to exchange information quickly, but it has also created new concerns regarding privacy and security.",

    "I wasn't particularly interested in the topic at first, but learning more about it made me realize that there were several interesting aspects I hadn't considered.",

    "The findings provide useful information that can contribute to future research and help researchers develop a more complete understanding of the issue.",

    "I made a few changes to my original plan after realizing that some of the steps would take longer than I had expected.",

    "Long-term success generally depends on the ability to adapt to changing conditions while maintaining a clear sense of purpose and direction.",

    "The main character gradually changes throughout the story as different experiences challenge the beliefs and assumptions that influenced earlier decisions.",

    "I checked the instructions one more time before turning in the assignment because I wanted to make sure I had answered every part of the question.",

    "Data-driven decision-making can help organizations identify patterns that might not be immediately visible through observation alone.",

    "The issue is complicated because there are several factors involved, and addressing only one of them may not produce a lasting solution.",

    "I felt more confident after practicing the material several times because I had a better idea of what I understood and what I still needed to review.",

    "The increasing use of automation has the potential to improve efficiency while also requiring workers to develop new skills and adapt to changing responsibilities.",

    "The conclusion is supported by the evidence presented throughout the analysis, although additional information could provide a more complete perspective.",

    "I decided to start earlier this time because waiting until the last minute had made the previous assignment much more stressful.",

    "Clear communication can reduce misunderstandings and help groups coordinate their efforts toward a common objective.",

    "The topic remains relevant because technological developments continue to create new questions about how society should approach innovation and responsibility.",

    "I realized that taking a few minutes to organize everything beforehand actually saved me time later because I didn't have to keep stopping to figure out what to do next.",

    "The overall effectiveness of a strategy depends on how well it addresses the specific circumstances and objectives involved.",

    "The information presented provides a useful foundation for understanding the issue, while also demonstrating why further discussion may be necessary.",

    "I didn't get the result I wanted on the first attempt, but looking at what went wrong helped me figure out what I should change next time."
    
    ]

labels = [0] * 166 + [1] * 254

# ==================================================
# TRAIN MODEL
# ==================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    min_df=2,
    max_df=0.9
)

X = vectorizer.fit_transform(texts)


print("TEXTS:", len(texts))
print("LABELS:", len(labels))

model = LogisticRegression(
    class_weight="balanced"
)

#model.fit(X, labels)

# ==================================================
# MODEL ACCURACY TEST
# ==================================================

test_texts = test_human + test_ai

test_labels = (
    [0] * len(test_human) +
    [1] * len(test_ai)
)

test_X = vectorizer.transform(test_texts)

test_predictions = model.predict(test_X)

correct_predictions = sum(
    prediction == actual
    for prediction, actual
    in zip(test_predictions, test_labels)
)

test_accuracy = round(
    (correct_predictions / len(test_labels)) * 100,
    1
)

st.write(f"TEST ACCURACY: {test_accuracy}%")

st.write(
    f"Correct: {correct_predictions} / {len(test_labels)}"
)

st.subheader("Incorrect Test Predictions")

for text, actual, prediction in zip(
    test_texts,
    test_labels,
    test_predictions
):

    if actual != prediction:

        actual_label = (
            "Human"
            if actual == 0
            else "AI"
        )

        predicted_label = (
            "Human"
            if prediction == 0
            else "AI"
        )

        st.write(
            f"**Actual:** {actual_label}  "
            f"**Predicted:** {predicted_label}"
        )

        st.write(text)

# ==================================================
# STREAMLIT UI
# ==================================================

st.title("AI Likelihood Classifier")

user_text = st.text_area(
    "Paste a paragraph"
)

ai_highlight_coverage = 0

# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button("Analyze", key="analyze_button") and user_text.strip():

    sample_X = vectorizer.transform([user_text])

    feature_names = vectorizer.get_feature_names_out()

    weights = model.coef_[0]

    input_vector = sample_X.toarray()[0]

    ai_reasons = []

    human_reasons = []

# ==================================================
# EXPLANATION ENGINE
# ==================================================

    for i, value in enumerate(input_vector):

        if value > 0:

            influence = value * weights[i]

            if influence > 0:

                ai_reasons.append(
                    (feature_names[i], influence)
                )

            elif influence < 0:

                human_reasons.append(
                    (feature_names[i], abs(influence))
                )

    ai_reasons.sort(
        key=lambda x: x[1],
        reverse=True
    )

    human_reasons.sort(
        key=lambda x: x[1],
        reverse=True
    )

    ai_reasons = ai_reasons[:5]

    human_reasons = human_reasons[:5]

    probabilities = model.predict_proba(sample_X)[0]

    human_probability = round(
        probabilities[0] * 100,
        1
    )

    ai_probability = round(
        probabilities[1] * 100,
        1
    )

    separation = abs(
        ai_probability -
        human_probability
    )

    evidence_strength = min(
        len(ai_reasons) +
        len(human_reasons),
        10
    ) * 10

    confidence = round(
        0.7 * separation +
        0.3 * evidence_strength,
        1
    )
 
# ==================================================
# RESULTS
# ==================================================

    sample_X = vectorizer.transform([user_text])

    probabilities = model.predict_proba(sample_X)[0]

    human_probability = round(
        probabilities[0] * 100,
        1
    )

    ai_probability = round(
        probabilities[1] * 100,
        1
    )

    # ==================================================
    # CATEGORY LOGIC
    # ==================================================

    if ai_probability >= 80:
        category = "Likely AI"

    elif ai_probability >= 60:
        category = "Possibly AI"

    elif ai_probability >= 40:
        category = "Uncertain"

    elif human_probability >= 60:
        category = "Possibly Human"

    elif human_probability >= 40:
        category = "Likely Human"

    else:
        category = "Very Uncertain"

    # ==================================================
    # RESULTS
    # ==================================================

    st.subheader("Results")

    st.write(f"Human Probability: {human_probability}%")

    st.write(f"AI Probability: {ai_probability}%")

    st.write(f"Confidence: {confidence}%")

    st.write(f"Category: {category}")

    # ==================================================
    # AI HIGHLIGHT COVERAGE
    # ==================================================

    import re

    sentences = re.split(
        r'(?<=[.!?])\s+',
        user_text.strip()
    )

    flagged_chars = 0
    total_chars = len(user_text)

    flagged_sentences = []

    for sentence in sentences:

        if not sentence.strip():
            continue

        sentence_X = vectorizer.transform([sentence])

        sentence_ai_probability = (
            model.predict_proba(sentence_X)[0][1] * 100
        )

        # Sentence is considered AI-like at 50%+
        if sentence_ai_probability >= 50:

            flagged_chars += len(sentence)

            flagged_sentences.append(
                (
                    sentence,
                    round(sentence_ai_probability, 1)
                )
            )

    if total_chars > 0:

        ai_highlight_coverage = round(
            (flagged_chars / total_chars) * 100,
            1
        )

    else:

        ai_highlight_coverage = 0

    # ==================================================
    # AI HIGHLIGHTING + REASONS
    # ==================================================

    import re
    import html

    st.subheader("AI Writing Analysis")

    sentences = re.split(
        r'(?<=[.!?])\s+',
        user_text.strip()
    )

    flagged_sentences = []
    highlighted_parts = []

    total_chars = len(user_text)
    highlighted_chars = 0

    for sentence in sentences:

        if not sentence.strip():
            continue

        sentence_X = vectorizer.transform([sentence])

        sentence_ai_probability = (
            model.predict_proba(sentence_X)[0][1] * 100
        )

        probability = round(
            sentence_ai_probability,
            1
        )

        safe_sentence = html.escape(sentence)

        # RED = strong AI signal (70%+)
        if probability >= 70:

            highlighted_parts.append(
                f'<span style="background-color:#ff9999; '
                f'padding:2px; border-radius:3px;">'
                f'{safe_sentence}</span>'
            )

            highlighted_chars += len(sentence)

            flagged_sentences.append(
                (
                    sentence,
                    probability,
                    "strong"
                )
            )

        # YELLOW = moderate AI signal (50–69.9%)
        elif probability >= 50:

            highlighted_parts.append(
                f'<span style="background-color:#fff29a; '
                f'padding:2px; border-radius:3px;">'
                f'{safe_sentence}</span>'
            )

            highlighted_chars += len(sentence)

            flagged_sentences.append(
                (
                    sentence,
                    probability,
                    "moderate"
                )
            )

        # NO HIGHLIGHT
        else:

            highlighted_parts.append(
                safe_sentence
            )

    # ==================================================
    # AI HIGHLIGHT COVERAGE
    # ==================================================

    if total_chars > 0:

        ai_highlight_coverage = round(
            (highlighted_chars / total_chars) * 100,
            1
        )

    else:

        ai_highlight_coverage = 0

    # ==================================================
    # DISPLAY HIGHLIGHTED PARAGRAPH
    # ==================================================

    st.markdown("### Highlighted Text")

    highlighted_text = " ".join(
        highlighted_parts
    )

    st.markdown(
        f"""
        <div style="
            padding:15px;
            border:1px solid #ddd;
            border-radius:8px;
            line-height:1.7;
            font-size:16px;
        ">
            {highlighted_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================================================
    # COVERAGE
    # ==================================================

    st.metric(
        "AI Highlight Coverage",
        f"{ai_highlight_coverage}%"
    )

    # ==================================================
    # LEGEND
    # ==================================================

    st.markdown(
        "🔴 **Strong AI signal (70%+)** &nbsp;&nbsp; "
        "🟡 **Moderate AI signal (50–69.9%)**"
    )
# ==================================================
# FLAGGED SENTENCES
# ==================================================

    st.subheader("Sentences Flagged")

    if flagged_sentences:

        for sentence, probability, strength in flagged_sentences:

            if strength == "strong":

                st.error(
                    f"🔴 **{probability}% AI**\n\n"
                    f"{sentence}"
                )

            else:

                st.warning(
                    f"🟡 **{probability}% AI**\n\n"
                    f"{sentence}"
                )

    else:

        st.write("No sentences were strongly flagged.")


# ==================================================
# EXISTING REASONS
# ==================================================

    with st.expander("Reasons for AI"):

        if ai_reasons:

            for word, score in ai_reasons:

                st.write(f"• {word}")

        else:

            st.write("None")


    with st.expander("Reasons for Human"):

        if human_reasons:

            for word, score in human_reasons:

                 st.write(f"• {word}")

        else:

            st.write("None")



