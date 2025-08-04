#-----------actions.py-----------------

from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, AllSlotsReset, ActiveLoop, SessionStarted, ActionExecuted, FollowupAction
from rasa_sdk.forms import FormValidationAction
import logging
import random
import httpx
from datetime import datetime
import hashlib
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Google Generative AI (optional)
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class EnhancedIntentClassifier:
    """Enhanced Intent Classifier that uses NLU examples first and LLM fallback"""
    
    def __init__(self):
        # self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.gemini_key = "AIzaSyCtD8WL1tzGXQtUBoirYAHIK8F21H4UQyM"
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        self.gemini_model = None
        self.use_gemini = False
        self.use_openai = False     
        
        # initialize gemini if available
        if GOOGLE_AI_AVAILABLE and self.google_api_key:
            try:
                genai.configure(api_key=self.gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("Google Generative AI (Gemini) initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google AI: {e}")
        else:
            logger.warning("GOOGLE_API_KEY not found or Google AI not available. LLM fallback disabled.")

            # Initialize OpenAI if available
        if OPENAI_AVAILABLE and self.openai_key:
            try:
                openai.api_key = self.openai_key
                self.use_openai = True
                logger.info("OpenAI API key found and client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

        if not self.use_gemini and not self.use_openai:
            logger.warning("No LLM API key found, LLM fallback is disabled.")
        
    def extract_amount_from_text(self, text: str) -> Optional[str]:
        """Extract amount from user input using regex patterns"""
        patterns = [
            r'(?:RM\s*)?(\d+(?:\.\d{1,2})?)\s*(?:ringgit|rm|MYR)?',
            r'(\d+(?:\.\d{1,2})?)\s*(?:ringgit|rm|MYR)',
            r'(?:top up|topup|add|deposit|load)\s*(?:RM\s*)?(\d+(?:\.\d{1,2})?)',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def is_topup_intent_llm(self, user_message: str) -> tuple[bool, float, Optional[str]]:
        """Use LLM to classify if the message is a topup intent"""
        if not self.model:
            return False, 0.0, None
        
        prompt = f"""
        Analyze this user message to determine if it's requesting a money topup/reload to their OWN account or wallet.
        User message: "{user_message}"
        
        Classification Rules:
        1. TOPUP Intent: User wants to add money to their OWN wallet/account from bank/FPX
        2. NOT TOPUP: Transfers to other people, payments, withdrawals, balance inquiries
        
        Examples of TOPUP:
        - "I want to topup my wallet"
        - "Add money to my account" 
        - "Load 100 into my wallet"
        - "Deposit money from bank"
        
        Examples of NOT TOPUP:
        - "Send money to John"
        - "Pay my bills"
        - "Check my balance"
        - "Withdraw money"
        
        Respond in this exact format:
        CLASSIFICATION: [TOPUP/NOT_TOPUP]
        CONFIDENCE: [0.0-1.0]
        AMOUNT: [extracted amount or NONE]
        REASONING: [brief explanation]
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            lines = response_text.split('\n')
            classification = None
            confidence = 0.0
            amount = None
            
            for line in lines:
                if line.startswith('CLASSIFICATION:'):
                    classification = line.split(':')[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    confidence = float(line.split(':')[1].strip())
                elif line.startswith('AMOUNT:'):
                    amount_str = line.split(':')[1].strip()
                    amount = amount_str if amount_str != 'NONE' else None
            
            is_topup = classification == 'TOPUP'
            return is_topup, confidence, amount
            
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return False, 0.0, None
    
    def classify_intent(self, user_message: str) -> Dict[str, Any]:
        """Main classification method"""
        # First try to extract amount using regex
        extracted_amount = self.extract_amount_from_text(user_message)
        
        # Check for topup keywords
        topup_keywords = ['topup', 'top up', 'add money', 'load money', 'deposit', 'reload', 'add funds']
        message_lower = user_message.lower()
        
        keyword_match = any(keyword in message_lower for keyword in topup_keywords)
        
        # If clear keyword match, return high confidence
        if keyword_match:
            return {
                'intent': 'topup_wallet',
                'confidence': 0.95,
                'extracted_amount': extracted_amount,
                'method': 'keyword_matching'
            }
        
        # Fallback to LLM if available
        if self.model:
            is_topup, llm_confidence, llm_amount = self.is_topup_intent_llm(user_message)
            if is_topup and llm_confidence > 0.7:
                return {
                    'intent': 'topup_wallet',
                    'confidence': llm_confidence,
                    'extracted_amount': llm_amount or extracted_amount,
                    'method': 'llm_classification'
                }
        
        return {
            'intent': 'unknown',
            'confidence': 0.0,
            'extracted_amount': extracted_amount,
            'method': 'no_match'
        }

# Initialize the enhanced classifier
enhanced_classifier = EnhancedIntentClassifier()

class ActionEnhancedIntentClassifier(Action):
    """Enhanced intent classification with LLM fallback"""
    
    def name(self) -> Text:
        return "action_enhanced_intent_classifier"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text', '')
        logger.info(f"Enhanced classification for: '{user_message}'")
        
        # Get classification result
        result = enhanced_classifier.classify_intent(user_message)
        
        logger.info(f"Classification result: {result}")
        
        events = []
        
        # If intent is topup_wallet, ensure amount is handled
        if result.get('intent') == 'topup_wallet':
            if result.get('extracted_amount'):
                try:
                    amount = float(result['extracted_amount'])
                    events.append(SlotSet("amount", amount))
                    logger.info(f"Set amount slot to: {amount}")
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert amount: {result['extracted_amount']}")
                    events.append(SlotSet("amount", None))
            else:
                # No amount provided, let the form handle it
                events.append(SlotSet("amount", None))
                logger.info("No amount provided, form will request it")
        
        return events
    
class ActionSetTransactionTypeTopup(Action):
    """Set transaction type to topup"""
    
    def name(self) -> Text:
        return "action_set_transaction_type_topup"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        return [SlotSet("transaction_type", "topup")]

class ValidateTopupFormEnhanced(FormValidationAction):
    """Enhanced validation for topup form"""
    
    def name(self) -> Text:
        return "validate_topup_form_enhanced"
    
    def validate_amount(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate amount slot"""
        if slot_value is None:
            dispatcher.utter_message(text="Please specify the amount you want to topup (e.g., RM 100).")
            return {"amount": None}
        
        try:
            amount = float(slot_value)
            if amount < 10:
                dispatcher.utter_message(text="❌ Minimum topup amount is RM 10.00")
                return {"amount": None}
            elif amount > 10000:
                dispatcher.utter_message(text="❌ Maximum topup amount is RM 10,000.00")
                return {"amount": None}
            else:
                return {"amount": amount}
        except (ValueError, TypeError):
            dispatcher.utter_message(text="❌ Please enter a valid amount (e.g., 100 or 50.50)")
            return {"amount": None}
    
    def validate_bank_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate bank name slot"""
        if slot_value is None:
            dispatcher.utter_message(text="Please select a bank from the available options.")
            return {"bank_name": None}
        
        valid_banks = ["SBI Bank", "ABC Bank", "Maybank", "CIMB Bank", "Public Bank"]
        
        # Check if the slot_value matches any valid bank (case insensitive)
        for bank in valid_banks:
            if bank.lower() == str(slot_value).lower():
                return {"bank_name": bank}
        
        dispatcher.utter_message(text="❌ Please select a valid bank from: SBI Bank, ABC Bank, Maybank, CIMB Bank, Public Bank.")
        return {"bank_name": None}
    
    def validate_fpx_username(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Ensure username is exactly 10 digits (string of numbers)"""
        val = str(slot_value).strip()
        if val.isdigit() and len(val) == 10:
            return {"fpx_username": val}
        dispatcher.utter_message(text="❌ Username must be exactly 10 digits (numbers only). Please try again.")
        return {"fpx_username": None}
    
    def validate_fpx_password(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Password must be at least 7 characters, can be any string"""
        val = str(slot_value)
        if len(val) >= 7:
            return {"fpx_password": val}
        dispatcher.utter_message(text="❌ Password must be at least 7 characters. Please try again.")
        return {"fpx_password": None}
    
    def validate_account_type(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        valid_types = ["Saving", "Current"]
        val = str(slot_value).capitalize()
        if val in valid_types:
            return {"account_type": val}
        dispatcher.utter_message(text="❌ Please select either 'Saving' or 'Current' account type.")
        return {"account_type": None}
    
    def validate_confirm_topup(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        if slot_value == "affirm":
            return {"confirm_topup": "affirm"}
        else:
            dispatcher.utter_message(text="Cancelled the topup as per your request.")
            return {"confirm_topup": "deny", "amount": None, "bank_name": None}
        
    def validate_confirm_final(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        if slot_value == "affirm":
            return {"confirm_final": "affirm"}
        else:
            dispatcher.utter_message(text="Topup cancelled at final confirmation step.")
            return {"confirm_final": "deny", "account_type": None}



class ValidateFpxAuthenticationForm(FormValidationAction):
    """Validation for FPX authentication form"""
    
    def name(self) -> Text:
        return "validate_fpx_authentication_form"
    
    def validate_fpx_username(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate FPX username"""
        if slot_value is None or len(str(slot_value).strip()) < 3:
            dispatcher.utter_message(text="❌ Username must be at least 3 characters long.")
            return {"fpx_username": None}
        
        return {"fpx_username": str(slot_value).strip()}
    
    def validate_fpx_password(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate FPX password"""
        if slot_value is None or len(str(slot_value).strip()) < 4:
            dispatcher.utter_message(text="❌ Password must be at least 4 characters long.")
            return {"fpx_password": None}
        
        return {"fpx_password": str(slot_value).strip()}

class ValidateAccountSelectionForm(FormValidationAction):
    """Validation for account selection form"""
    
    def name(self) -> Text:
        return "validate_account_selection_form"
    
    def validate_account_type(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate account type"""
        if slot_value is None:
            return {"account_type": None}
        
        valid_types = ["savings", "current"]
        slot_value_lower = str(slot_value).lower()
        
        if slot_value_lower in valid_types:
            return {"account_type": slot_value_lower}
        elif "saving" in slot_value_lower:
            return {"account_type": "savings"}
        elif "current" in slot_value_lower:
            return {"account_type": "current"}
        else:
            dispatcher.utter_message(text="❌ Please select either 'savings' or 'current' account.")
            return {"account_type": None}

class ActionAuthenticateFPX(Action):
    """Simulate FPX authentication"""
    
    def name(self) -> Text:
        return "action_authenticate_fpx"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        username = tracker.get_slot("fpx_username")
        password = tracker.get_slot("fpx_password")
        
        # Simulate authentication logic
        # For demo purposes, some usernames will fail
        failing_usernames = ["wronguser", "invaliduser", "testfail"]
        
        if username and username.lower() in failing_usernames:
            return [SlotSet("fpx_authenticated", False)]
        elif username and password and len(username) >= 3 and len(password) >= 4:
            return [SlotSet("fpx_authenticated", True)]
        else:
            return [SlotSet("fpx_authenticated", False)]

class ActionExecuteTopup(Action):
    """Execute the topup transaction"""
    
    def name(self) -> Text:
        return "action_execute_topup"
    
    def _generate_transaction_reference(self) -> str:
        """Generate a unique transaction reference"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = str(random.randint(1000, 9999))
        return f"TXN{timestamp}{random_part}"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        amount = tracker.get_slot("amount")
        bank_name = tracker.get_slot("bank_name")
        account_type = tracker.get_slot("account_type")
        
        # Generate transaction reference
        transaction_ref = self._generate_transaction_reference()
        
        # Simulate different transaction outcomes
        # For demo purposes, randomly assign outcomes
        outcome = random.choice(["completed", "completed", "completed", "pending", "failed"])
        
        # Some specific cases for demo
        if amount and amount > 5000:
            outcome = "pending"  # Large amounts might be pending
        elif bank_name == "ABC Bank":
            outcome = random.choice(["completed", "failed"])  # ABC Bank sometimes fails
        
        logger.info(f"Transaction outcome: {outcome} for amount: {amount}")
        
        events = [
            SlotSet("transaction_reference", transaction_ref),
            SlotSet("transaction_status", outcome)
        ]
        
        return events

class ActionResetSlots(Action):
    """Reset all transaction-related slots"""
    
    def name(self) -> Text:
        return "action_reset_slots"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        return [
            SlotSet("transaction_type", None),
            SlotSet("amount", None),
            SlotSet("bank_name", None),
            SlotSet("account_type", None),
            SlotSet("fpx_username", None),
            SlotSet("fpx_password", None),
            SlotSet("fpx_authenticated", False),
            SlotSet("transaction_status", None),
            SlotSet("transaction_reference", None)
        ]

class ActionSessionStart(Action):
    """Custom session start action"""
    
    def name(self) -> Text:
        return "action_session_start"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Initialize session with clean slate
        events = [SessionStarted()]
        
        # Add a follow-up action to greet the user
        events.append(ActionExecuted("utter_greet"))
        
        return events

class ActionDefaultFallback(Action):
    """Default fallback action"""
    
    def name(self) -> Text:
        return "action_default_fallback"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Check if user might be trying to do a topup
        user_message = tracker.latest_message.get('text', '').lower()
        topup_hints = ['money', 'topup', 'add', 'load', 'deposit', 'wallet', 'bank', 'ringgit', 'rm']
        
        if any(hint in user_message for hint in topup_hints):
            dispatcher.utter_message(
                text="It seems like you want to topup your wallet. Try saying:\n"
                     "• 'I want to topup RM 100'\n"
                     "• 'Add money to my wallet'\n"
                     "• 'Load 50 ringgit'"
            )
        else:
            dispatcher.utter_message(text="I didn't understand that. Can you please rephrase?")
        
        return []

# Additional utility actions

class ActionCheckBalance(Action):
    """Mock action to check wallet balance"""
    
    def name(self) -> Text:
        return "action_check_balance"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Mock balance for demo
        balance = random.uniform(100, 5000)
        
        dispatcher.utter_message(
            text=f"💰 Your current wallet balance is RM {balance:.2f}"
        )
        
        return []

class ActionShowHelp(Action):
    """Show detailed help information"""
    
    def name(self) -> Text:
        return "action_show_help"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        help_text = """
🤖 **I can help you with:**

💰 **Wallet Topup**
- Add money to your wallet from bank account
- Supported banks: SBI, ABC, Maybank, CIMB, Public Bank
- Amount range: RM 10 - RM 10,000

📝 **How to use:**
- Say: "Topup RM 100" or "Add money to wallet"
- Follow the prompts for bank selection
- Provide FPX credentials for authentication
- Choose savings or current account

🔒 **Security:**
- All transactions use secure FPX authentication
- Your credentials are not stored
- Transaction references provided for all operations

❓ **Need help?** Just ask me!
        """
        
        dispatcher.utter_message(text=help_text)
        
        return []