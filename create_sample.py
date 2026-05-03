from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def create_invoice():
    c = canvas.Canvas("sample_documents/invoice_test.pdf", pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, h-60, "INVOICE")

    c.setFont("Helvetica", 12)
    c.drawString(50,  h-100, "Invoice Number : INV-2024-042")
    c.drawString(50,  h-120, "Issue Date     : 15 October 2024")
    c.drawString(50,  h-140, "Due Date       : 14 November 2024")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50,  h-180, "From:")
    c.setFont("Helvetica", 12)
    c.drawString(50,  h-200, "Mueller GmbH")
    c.drawString(50,  h-220, "Hauptstrasse 12, 60311 Frankfurt am Main")
    c.drawString(50,  h-240, "VAT ID: DE123456789")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50,  h-280, "Bill To:")
    c.setFont("Helvetica", 12)
    c.drawString(50,  h-300, "Tech Solutions AG")
    c.drawString(50,  h-320, "Berliner Allee 5, 40212 Duesseldorf")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50,  h-370, "Description")
    c.drawString(300, h-370, "Qty")
    c.drawString(370, h-370, "Unit Price")
    c.drawString(460, h-370, "Total")
    c.line(50, h-378, 540, h-378)

    c.setFont("Helvetica", 11)
    c.drawString(50,  h-395, "Software Development Services")
    c.drawString(300, h-395, "10")
    c.drawString(370, h-395, "EUR 350.00")
    c.drawString(460, h-395, "EUR 3,500.00")

    c.drawString(50,  h-415, "System Architecture Consulting")
    c.drawString(300, h-415, "3")
    c.drawString(370, h-415, "EUR 440.00")
    c.drawString(460, h-415, "EUR 1,320.00")

    c.line(50, h-430, 540, h-430)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(370, h-450, "Subtotal:")
    c.drawString(460, h-450, "EUR 4,820.00")
    c.drawString(370, h-470, "VAT (19%):")
    c.drawString(460, h-470, "EUR 915.80")
    c.drawString(370, h-495, "TOTAL DUE:")
    c.drawString(460, h-495, "EUR 5,735.80")

    c.setFont("Helvetica", 10)
    c.drawString(50, h-550, "Payment Terms: 30 days net")
    c.drawString(50, h-570, "Bank: Deutsche Bank | IBAN: DE89 3704 0044 0532 0130 00")

    c.save()
    print("Sample invoice created: sample_documents/invoice_test.pdf")

create_invoice()