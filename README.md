# Getting Started

1. Install **Pipenv** if not in already: `pip install pipenv`
2. Run `pipenv install` to create a virtual environment and get all deps set up
3. You should now be able to run the `.py` files normally!

---

![logo](./github/kanjo-logo.gif)

**Games make us feel a lot of things.**

We're usually too busy paying attention to what's going on than to ourselves, though!

*Wouldn't it be cool to see the emotional rollercoasters you go through when playing games?*

## How It Works

Headless OpenCV program begins tracking your face when you play a game.

- Would be run locally for security purposes
- It detects what you're feeling based on your expression at different timepoints
- After playing the game, you get a graph summarizing your emotions throughout gameplay

*Loads of other cool stuff can be done with this data!*

### 😡 Rage Detection
Detection of emotions like anger could warn you to stop mindlessly throwing yourself against a Dark Souls boss with absolutely no return (because you're angry to play correctly :))

### 🤖 Automation
When high amounts of an emotion are detected, trigger some kind of event/script
> e.g. You're about to hit a quad-feed but you're too locked in to start recording, so [the app] detects your concentration and begins recording for you

### 📊 Interesting Statistics
Showing you how your emotions fluctuate based on what game you play, what genre, etc.
> e.g. How many minutes into your Elden Ring session are you usually the angriest? Calmest?
- Showing you how other people's emotions fluctuate in a similar fashion
- Showing you how your emotions look like after you progress a certain amount in a game
