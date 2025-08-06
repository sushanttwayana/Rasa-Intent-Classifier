# -------------------- actions.py --------------------

from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, AllSlotsReset, ActiveLoop, SessionStarted, ActionExecuted
from rasa_sdk.forms import FormValidationAction
import logging
import random
import httpx
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidateTopupForm(FormValidationAction):
    """
    This validates the form for the topup
    """
    
    def name(self) -> Text:
        return "validate_topup_form"
    
    async def validate_amount(
        self, value: Text, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        
        """
        Validate amount is positive and within limits.
        """

        try:
            amount = float(value)
            
            if amount <= 0:
                dispatcher.utter_message(text="Amount must be greater than zero.")
                return {"amount": None}
                
            elif amount > 10000: # MYR topup limit
                dispatcher.utter_message(text = "Maximum topup amount is 10,000 MYR")
                return {"amount": None}

            return {"amount": value}

        except ValueError:
            dispatcher.utter_message(text="Please enter a valid amount (e.g. 100 or 50.50)")
            return {"amount": None}
        
    async def validate_recipient_type(
            self,
            value: Text,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
        ) -> Dict[Text, Any]:
            """Validate supported wallet types."""
            
            # Ensure these values align with domain.yml recipient_type values
            valid_wallets = ["mobile_wallet", "bank_account", "own_account", "business_account", "generic_contact", "merchant"]
            
            # Check if the extracted value is one of the specific mobile wallet types you support.
            # You might need more sophisticated entity extraction or a separate custom slot for sub-types
            # For now, let's assume the NLU extracts one of the broad recipient_type categories.
            # If "Touch 'n Go" is extracted as `recipient_type`, Rasa will map it to "mobile_wallet" via NLU or rules.
            # So, validate based on the canonical values.
            
            if value.lower() not in valid_wallets:
                dispatcher.utter_message(text=f"We support types like {', '.join(valid_wallets)}. Please specify.")
                return {"recipient_type": None}
            return {"recipient_type": value}
    
class ValidateRemittanceForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_remittance_form"

    async def validate_target_country(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        valid_countries = ["Nepal", "India"]
        if value not in valid_countries:
            dispatcher.utter_message(text=f"We only support remittance to {', '.join(valid_countries)}")
            return {"target_country": None}
        return {"target_country": value}        

class ValidateTransferForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_transfer_form"

    async def validate_recipient_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Verify recipient name format."""
        if len(value) < 3:
            dispatcher.utter_message(text="Recipient name too short. Please provide a full name.")
            return {"recipient_name": None}
        return {"recipient_name": value}
    
class ActionCalculateConversion(Action):
    def name(self) -> Text:
        return "action_calculate_conversion"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        amount = float(tracker.get_slot("amount") or 0)
        target_country = tracker.get_slot("target_country")
        
        # Ensure exchange_rate is fetched/set before calling this if it's dynamic
        # For now, using the hardcoded rates as a fallback/initial source
        
        rates = {
            "Nepal": {"rate": 32.45, "fee": 5.00},
            "India": {"rate": 18.20, "fee": 8.00}
        }
        
        exchange_rate = rates.get(target_country, {}).get("rate")
        remittance_fee = rates.get(target_country, {}).get("fee")

        if amount and exchange_rate and remittance_fee is not None: # Check for None for fee
            converted_amount = amount * exchange_rate
            logger.info(f"Calculated: {amount} MYR to {converted_amount:.2f} in {target_country} with rate {exchange_rate} and fee {remittance_fee}")
            return [
                SlotSet("converted_amount", converted_amount),
                SlotSet("exchange_rate", exchange_rate),
                SlotSet("remittance_fee", remittance_fee)
            ]
        else:
            logger.warning(f"Unable to calculate conversion: amount={amount}, target_country={target_country}, exchange_rate={exchange_rate}, remittance_fee={remittance_fee}")
            dispatcher.utter_message(text="Unable to calculate conversion. Please ensure amount and target country are provided and supported.")
            return []
    
class ActionExecuteTopup(Action):
    
    def name(self) -> Text:
        return "action_execute_topup"
    
    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        # Extract slots
        amount = tracker.get_slot("amount")
        recipient_name = tracker.get_slot("recipient_name") # Use recipient_name for consistency
        recipient_type = tracker.get_slot("recipient_type") # Use recipient_type for consistency
        currency = tracker.get_slot("currency") or "MYR" # Default to MYR
        
        # Ensure values are not None before proceeding
        if amount is None or recipient_name is None or recipient_type is None:
            logger.error("Missing slots for topup execution.")
            dispatcher.utter_message(text="I'm missing some details to complete the top-up. Please try again.")
            return [AllSlotsReset()] # Reset and restart
        
        ref_id = f"TOP-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

        try:
            # Simulate API call - Replace with actual banking API
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(
            #         "https://api.yourbank.com/topup",
            #         json={
            #             "amount": amount,
            #             "currency": currency,
            #             "recipient_name": recipient_name,
            #             "recipient_type": recipient_type,
            #             "reference": ref_id
            #         },
            #         timeout=10.0
            #     )
            #     response.raise_for_status()
            #     result = response.json()

            # Simulated success response
            result = {"success": True, "reference": ref_id}

            if result.get("success"):
                logger.info(f"Topup successful: {amount} {currency} to {recipient_name} ({recipient_type})")
                dispatcher.utter_message(
                    response="utter_topup_success",
                    amount=f"{amount:.2f}",
                    currency=currency,
                    recipient_name=recipient_name, # Pass correct slot name
                    recipient_type=recipient_type,
                    transaction_reference=ref_id # Pass correct slot name
                )
                return [AllSlotsReset(), ActionExecuted("action_listen")] # Ensure listen after reset
            
            dispatcher.utter_message(text="Topup failed. Please try again later.")
            return [AllSlotsReset(), ActionExecuted("action_listen")]

        except Exception as e:
            logger.error(f"Topup error: {str(e)}")
            dispatcher.utter_message(text="Technical error occurred during topup. Please try again later or contact support.")
            return [AllSlotsReset(), ActionExecuted("action_listen")]
        

class ActionExecuteRemittance(Action):
    def name(self) -> Text:
        return "action_execute_remittance"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        amount = tracker.get_slot("amount")
        target_country = tracker.get_slot("target_country")
        converted_amount = tracker.get_slot("converted_amount")
        recipient_name = tracker.get_slot("recipient_name") # Use recipient_name
        recipient_type = tracker.get_slot("recipient_type")
        payment_method = tracker.get_slot("payment_method")
        transaction_purpose = tracker.get_slot("transaction_purpose")

        # Ensure all necessary slots are available
        if any(s is None for s in [amount, target_country, converted_amount, recipient_name, recipient_type, payment_method, transaction_purpose]):
            logger.error("Missing slots for remittance execution.")
            dispatcher.utter_message(text="I'm missing some details to complete the remittance. Please try again.")
            return [AllSlotsReset(), ActionExecuted("action_listen")]

        ref_id = f"REM-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

        try:
            # Simulate remittance API call
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(
            #         "https://api.yourbank.com/remit",
            #         json={
            #             "source_amount": amount,
            #             "source_currency": "MYR",
            #             "target_currency": "NPR" if target_country == "Nepal" else "INR", # Assuming INR for India
            #             "converted_amount": converted_amount,
            #             "recipient_name": recipient_name,
            #             "recipient_type": recipient_type,
            #             "payment_method": payment_method,
            #             "transaction_purpose": transaction_purpose,
            #             "reference": ref_id
            #         }
            #     )
            #     response.raise_for_status()
            #     result = response.json()

            # Simulated success
            result = {"success": True, "reference": ref_id}

            if result.get("success"):
                dispatcher.utter_message(
                    response="utter_remittance_success",
                    amount=f"{amount:.2f}",
                    converted_amount=f"{converted_amount:.2f}",
                    recipient_name=recipient_name, # Pass correct slot name
                    target_country=target_country,
                    transaction_reference=ref_id # Pass correct slot name
                )
                return [AllSlotsReset(), ActionExecuted("action_listen")]

            dispatcher.utter_message(text="Remittance failed. Please try again later.")
            return [AllSlotsReset(), ActionExecuted("action_listen")]

        except Exception as e:
            logger.error(f"Remittance error: {str(e)}")
            dispatcher.utter_message(text="Technical error occurred during remittance. Please visit a branch or try again.")
            return [AllSlotsReset(), ActionExecuted("action_listen")]
        
class ActionFetchExchangeRate(Action):
    def name(self) -> Text:
        return "action_fetch_exchange_rate"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        target_country = tracker.get_slot("target_country")
        
        if not target_country:
            dispatcher.utter_message(text="Which country are you interested in for exchange rates?")
            return [] # Keep target_country slot requesting

        try:
            # Simulate rate API - Replace with real API call
            rates = {
                "Nepal": {"rate": 32.45, "fee": 5.00},
                "India": {"rate": 18.20, "fee": 8.00}
            }
            
            if target_country in rates:
                return [
                    SlotSet("exchange_rate", rates[target_country]["rate"]),
                    SlotSet("remittance_fee", rates[target_country]["fee"])
                ]
            
            dispatcher.utter_message(text=f"No rates available for {target_country}.")
            return []
            
        except Exception as e:
            logger.error(f"Rate fetch error: {str(e)}")
            dispatcher.utter_message(text="Couldn't fetch exchange rates due to a technical issue. Please try again later.")
            return [SlotSet("exchange_rate", None), SlotSet("remittance_fee", None)] # Set to None on error

class ActionExecuteTransfer(Action):
    def name(self) -> Text:
        return "action_execute_transfer"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        amount = tracker.get_slot("amount")
        recipient_name = tracker.get_slot("recipient_name")
        recipient_type = tracker.get_slot("recipient_type")
        currency = tracker.get_slot("currency") or "MYR"
        payment_method = tracker.get_slot("payment_method")
        transaction_purpose = tracker.get_slot("transaction_purpose")

        # Ensure all necessary slots are available
        if any(s is None for s in [amount, recipient_name, recipient_type, currency, payment_method, transaction_purpose]):
            logger.error("Missing slots for transfer execution.")
            dispatcher.utter_message(text="I'm missing some details to complete the transfer. Please try again.")
            return [AllSlotsReset(), ActionExecuted("action_listen")]

        ref_id = f"TXN-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

        try:
            # Simulate transfer API call
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(
            #         "https://api.yourbank.com/transfer",
            #         json={
            #             "amount": amount,
            #             "currency": currency,
            #             "recipient_name": recipient_name,
            #             "recipient_type": recipient_type,
            #             "payment_method": payment_method,
            #             "transaction_purpose": transaction_purpose,
            #             "reference": ref_id
            #         }
            #     )
            #     response.raise_for_status()
            #     result = response.json()

            # Simulated success
            result = {"success": True, "reference": ref_id}

            if result.get("success"):
                logger.info(f"Transfer successful: {amount} {currency} to {recipient_name} ({recipient_type})")
                dispatcher.utter_message(
                    response="utter_transfer_success",
                    amount=f"{amount:.2f}",
                    currency=currency,
                    recipient_name=recipient_name, # Corrected slot name
                    recipient_type=recipient_type,
                    transaction_reference=ref_id # Corrected slot name
                )
                return [AllSlotsReset(), ActionExecuted("action_listen")]
            
            dispatcher.utter_message(text="Transfer failed. Please try again later.")
            return [AllSlotsReset(), ActionExecuted("action_listen")]

        except Exception as e:
            logger.error(f"Transfer error: {str(e)}")
            dispatcher.utter_message(text="Technical error occurred during transfer. Please try again later or contact support.")
            return [AllSlotsReset(), ActionExecuted("action_listen")]


class ActionResetSlots(Action):
    def name(self) -> Text:
        return "action_reset_slots"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        logger.info("Resetting all slots and active loop.")
        return [AllSlotsReset(), ActiveLoop(None), ActionExecuted("action_listen")] # Ensure listening after reset

class ActionSetTransactionTypeTopup(Action):
    def name(self) -> Text:
        return "action_set_transaction_type_topup"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        logger.info("Setting transaction_type to 'topup'")
        return [SlotSet("transaction_type", "topup")]

class ActionSetTransactionTypeTransfer(Action):
    def name(self) -> Text:
        return "action_set_transaction_type_transfer"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        logger.info("Setting transaction_type to 'transfer'")
        return [SlotSet("transaction_type", "transfer")]

class ActionSetTransactionTypeRemittance(Action):
    def name(self) -> Text:
        return "action_set_transaction_type_remittance"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        logger.info("Setting transaction_type to 'remittance'")
        return [SlotSet("transaction_type", "remittance")]

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        logger.info("Session started. Resetting slots and listening.")
        # the session should begin with a `session_started` event
        events = [SessionStarted()]

        # any slots that should be carried over should come after the `SessionStarted` event
        # this is necessary to ensure the actions that are executed after the `session_started`
        # event have access to the slots that have been set before the session starts
        # events.extend(tracker.slots_as_entities()) # Use this if you want to carry over slots

        # an `action_listen` should be added at the end
        events.append(ActionExecuted("action_listen"))

        return events