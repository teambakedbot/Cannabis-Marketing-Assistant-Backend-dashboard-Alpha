from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from ..services.auth_service import get_firebase_user, get_current_user_optional
import os
from ..crud.crud import (
    create_order,
)
from ..models.schemas import (
    User,
    OrderRequest,
)
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from twilio.rest import Client
from ..config.config import settings

# from fastapi.middleware.throttle import ThrottleMiddleware


router = APIRouter(
    prefix="/api/v1",
    tags=["order"],
    responses={404: {"description": "Not found"}},
)


def send_sms(to_phone: str, body: str):
    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body, from_=settings.TWILIO_PHONE_NUMBER, to=to_phone
        )
        print(f"SMS sent. SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False


def send_email(to_email: str, subject: str, body: str):
    message = Mail(
        from_email=settings.SENDGRID_FROM_EMAIL,
        to_emails=to_email,
        subject=subject,
        html_content=body,
    )
    try:
        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"Email sent to {to_email}. Status Code: {response.status_code}")
        return True
    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")
        return False


@router.post("/checkout")
async def place_order(
    order: OrderRequest,
    current_user: User = Depends(get_current_user_optional),
):
    print(f"Placing order: {order}")
    try:
        # Create order in the database
        new_order = await create_order(order)

        # Function to format cart items into HTML table
        def format_cart_items(cart):
            table_html = """
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 12px; border: 1px solid #dee2e6; text-align: left;">SKU</th>
                        <th style="padding: 12px; border: 1px solid #dee2e6; text-align: left;">Product Name</th>
                        <th style="padding: 12px; border: 1px solid #dee2e6; text-align: center;">Weight</th>
                        <th style="padding: 12px; border: 1px solid #dee2e6; text-align: center;">Quantity</th>
                    </tr>
                </thead>
                <tbody>
            """

            for item in cart.values():
                weight = item["weight"] if item["weight"] else "N/A"
                table_html += f"""
                    <tr>
                        <td style="padding: 12px; border: 1px solid #dee2e6;">{item['sku']}</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6;">{item['product_name']}</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; text-align: center;">{weight}</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; text-align: center;">{item['quantity']}</td>
                    </tr>
                """

            table_html += """
                </tbody>
            </table>
            """
            return table_html

        # Prepare customer email content
        customer_subject = "Order Confirmation"
        customer_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2>Dear {order.name},</h2>

            <p>Thank you for your order. We have received the following details:</p>

            <ul>
                <li><strong>Order ID:</strong> {new_order.id}</li>
                <li><strong>Name:</strong> {order.name}</li>
                <li><strong>Email:</strong> {order.contact_info.email}</li>
                <li><strong>Phone:</strong> {order.contact_info.phone or 'Not provided'}</li>
            </ul>

            <h3>Order Details:</h3>
            {format_cart_items(order.cart)}

            <p>We will contact you soon with pickup details.</p>

            <p>Best regards,<br>BakedBot</p>
        </body>
        </html>
        """

        # Prepare retailer email content
        retailer_subject = "New Order Received"
        retailer_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2>New Order Received</h2>

            <p>A new order has been placed with the following details:</p>

            <ul>
                <li><strong>Order ID:</strong> {new_order.id}</li>
                <li><strong>Customer Name:</strong> {order.name}</li>
                <li><strong>Email:</strong> {order.contact_info.email}</li>
                <li><strong>Phone:</strong> {order.contact_info.phone or 'Not provided'}</li>
            </ul>

            <h3>Order Details:</h3>
            {format_cart_items(order.cart)}

            <p>Please process this order as soon as possible.</p>
        </body>
        </html>
        """
        customer_sms_sent = False
        customer_email_sent = False
        retailer_email_sent = False
        # Send emails
        if order.contact_info.phone is not None:
            customer_sms_sent = send_sms(
                order.contact_info.phone, f"New order received: {new_order.id}"
            )

        if order.contact_info.email is not None:
            customer_email_sent = send_email(
                order.contact_info.email, customer_subject, customer_body
            )

        retailer_email_sent = send_email(
            settings.RETAILER_EMAIL, retailer_subject, retailer_body
        )

        if (customer_sms_sent or customer_email_sent) and retailer_email_sent:
            return {
                "message": "Order placed successfully and confirmation emails sent."
            }
        elif customer_sms_sent or customer_email_sent:
            return {
                "message": "Order placed successfully and customer email sent, but there was an issue sending the retailer email."
            }
        elif retailer_email_sent:
            return {
                "message": "Order placed successfully and retailer email sent, but there was an issue sending the customer email."
            }
        else:
            return {
                "message": "Order placed successfully, but there were issues sending confirmation emails."
            }

    except Exception as e:
        print(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
