import random
import re

class DiceRoller:
    def __init__(self, notation, num_rolls=1):
        self.notation = notation
        self.num_rolls = num_rolls
        # Updated regex: (\d*) makes the leading digit optional (defaults to 1)
        self.dice_pattern = re.compile(r"(\d*)\s*d\s*(\d+)(?:\s*k\s*(\d+))?", re.IGNORECASE)

    def roll_dice(self):
        # Debug: print what we're receiving
        print(f"DEBUG: Received notation: '{self.notation}' (type: {type(self.notation)})")
        
        # Try to extract dice notation from the input
        # Use search instead of match to find the pattern anywhere in the string
        match = self.dice_pattern.search(self.notation)
        
        if not match:
            raise ValueError(f"Invalid dice notation: '{self.notation}'. Expected format like 'd6', '1d6', '2d20k1', etc.")
        print(f"DEBUG: Match groups: {match.groups()}")

        # Group 1: number of dice (optional, defaults to 1)
        num_dice = int(match.group(1)) if match.group(1) else 1
        # Group 2: number of sides
        dice_sides = int(match.group(2))
        # Group 3: number to keep (optional, defaults to num_dice)
        keep = int(match.group(3)) if match.group(3) else num_dice

        rolls = [random.randint(1, dice_sides) for _ in range(num_dice)]
        rolls.sort(reverse=True)
        kept_rolls = rolls[:keep]

        return rolls, kept_rolls

    def roll_multiple(self):
        """Roll the dice multiple times according to num_rolls"""
        results = []
        for _ in range(self.num_rolls):
            rolls, kept_rolls = self.roll_dice()
            results.append({
                "rolls": rolls,
                "kept": kept_rolls,
                "total": sum(kept_rolls)
            })
        return results

    def __str__(self):
        if self.num_rolls == 1:
            rolls, kept_rolls = self.roll_dice()
            return f"ROLLS: {', '.join(map(str, rolls))} -> RETURNS: {sum(kept_rolls)}"
        else:
            results = self.roll_multiple()
            result_strs = []
            for i, result in enumerate(results, 1):
                result_strs.append(f"Roll {i}: ROLLS: {', '.join(map(str, result['rolls']))} -> RETURNS: {result['total']}")
            return "\n".join(result_strs)

if __name__ == "__main__":
    notation = input("Enter dice notation (e.g., 2d20k1): ")
    num_rolls = int(input("Number of rolls: ") or "1")
    dice_roller = DiceRoller(notation, num_rolls)
    print(dice_roller) 