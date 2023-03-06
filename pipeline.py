from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch
from tqdm import tqdm


morph_pairs = [["a cat", "a dog"], ["a cat", "a bunny"], ["a dog", "a bunny"], ["a whale", "a shark"],
               ["a hammerhead shark", "a tiger shark"], ["a cock", "a hen"], ["a leatherback turtle", "a box turtle"],
               ["a dog", "a hyena"], ["a cat", "a lion"], ["a cat","a tiger"], ["a lion", "tiger"], ["an alligator", "a lizard"],
               ["a scorpion", "a spider"], ["a dog" , "a wolf"], ["a peacock", "a quail"], ["a goose", "a duck"], ["a chicken" , "a duck"],
               ["a goose", "a swang"], ["a bear", "a koala"], ["a bear", "a kangaroo"], ["a dungeon crab", "a king crab"],
               ["a lobster", "a crawfish"], ["a spoonbill", "a flamingo"], ["a flamingo" , "a pelican"],["a fox", "a hyena"],
               ["a dog", "a fox"], ["a fly" , "a bee"], ["a bee", "a butterfly"], ["a starfish", "a sea urchin"],
               ["a hamster", "a hedgehog"], ["a groundhog", "a beaver"], ["a horse", "a zebra"], ["a guinea pig", "a chinchilla"],
               ["a buffalo", "a bison"], ["a worthog", "a pig"],["a weasal", "a mink"], ["a skunk", "a badger"],
               ["a monkey", "a gorilla"],["a panda", "a black bear"],["a soccer", "a basketball"],["a basketball", "a volleyball"],
               ["a guitar", "a violin"], ["a cabinet", "a dresser"],["a cleaver", "a knife"], ["a coffee mug", "a glass"],
               ["a cowboy boot", "a sneaker"], ["guitar", "electric guitar"], ["a wooden chair", "a folding chair"],
               ["a TV remote", "a phone"], ["a truck", "a garbage truck"], ["an ipod" , "an iphone"], ["a jeep", "an SUV"],
               ["a boat", "a lifeboat"], ["a shuttle" , "a minivan"], ["a monitor", "a TV"], ["a paint brush", "a toothbrush"],
               ["a car", "a police car"], ["a car", "a race car"], ["a reflex camera", "a digital camera"],
               ["a shuttle", "a school bus"], ["a pingpong ball", "a tennis ball"],["a truck", "a tow truck"],
               ["a keyboard", "a typewritter keyboard"], ["a bicycle", "a mortorcycle"], ["a vending machine", "an ATM"],
               ["a water bottle", "a wine bottle"], ["a cheeseburger", "a chicken sandwich"], ["a cauliflower", "a brocolli"],
               ["a spaghetti squash", "a butternut squash"], ["a strawberry" , "a blueberry"], ["a strawberry" , "a blackberry"],
               ["a blueberry", "a blackberry"], ["an orange", "a strawberry"], ["an orange", "a lemon"],
               ["an apple", "a pear"], ["an apple", "an orange"], ["a pizza", "a potpie"], ["a mountain", "a volcano"]]

result_save_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/stable_diffusion/03_04_2023_78_pairs"

for i in tqdm(range(len(morph_pairs))):
    one_pair = morph_pairs[i]

    source = one_pair[0]
    target = one_pair[1]

    morph_name = source.split(" ", 1)[-1] + "2" + target.split(" ", 1)[-1]

    pipeline = StableDiffusionWalkPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16",
    ).to("cuda")

    video_path = pipeline.walk(
        prompts=one_pair,
        seeds=[42, 1337],
        num_interpolation_steps=300,
        height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
        width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
        output_dir=result_save_path,        # Where images/videos will be saved
        name=morph_name,        # Subdirectory of output_dir where images/videos will be saved
        guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
        num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
    )