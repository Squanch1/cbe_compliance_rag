# FAQ 源材料：高频问答清单

## 这份文档是什么

FAQ 直出链路（`docs/adr/0001-faq-direct-answer.md`）需要一份中文问答表。表里的**问题**由这份文档给出，**答案**需要翻译——本文档提供待翻译的英文原文，翻译交给外部大模型，提示词见 `translation-prompt.md`。

**问题已经写成中文**，不用再翻。理由：问题是 BM25 的匹配键，必须用卖家实际会用的说法和术语（「进口一站式服务(IOSS)」而不是「进口一站式服务系统」），这层判断在整理阶段做比在翻译阶段做靠谱。

## 怎么用

1. 把每条下面的 **English answer** 交给外部大模型，用 `translation-prompt.md` 的提示词翻译
2. 译完的中文与英文原文**一并入库**（双语对照，见 ADR 2.4 节）
3. 问题直接用本文档的 `问题` 字段

## 出处标注说明

- `欧盟指南` 指 `eu-vat-ecommerce-explanatory-notes-rev-2027.pdf`（生效日 2027-01-01）。**两版问句相同**，取新版是因为它包含了 2027 年 ViDA 包的修订；2020 版与它不一致的地方已在 `conflict` 类评测用例里单独覆盖。
- `Amazon 政策` 指 `amazon-eu-vat-faq.html`。
- 章节号是该 PDF 的编号，用于核对与定位。

## 一、待翻译（欧盟官方指南，英文）

### 第一组：IOSS 是什么，值不值得用

---

**F-01**　问题：用 IOSS 发货给欧盟买家，对我有什么好处？

出处：欧盟指南 4.2.10 (a) 问题 1

English answer:

By using the IOSS, a supplier or an electronic interface (deemed supplier) ensures a transparent transaction for the customer who pays a VAT inclusive price at the time of the online purchase. The customer has certainty about the total price of the transaction and is not confronted with unexpected costs (VAT and, in principle, additional clearance fee) to be paid when goods are imported into the EU.

Furthermore, the use of the IOSS aims for a quick release of the goods by the customs authorities and a speedy delivery of the goods to the customer, which is often crucial for the latter.

The use of the IOSS also simplifies logistics as the goods can enter the EU and be released for free circulation in any Member State, regardless as to which Member State they are ultimately destined for.

---

**F-02**　问题：注册 IOSS 有没有营业额门槛？

出处：欧盟指南 4.2.10 (b) 问题 4

English answer:

No, there is no threshold to register in the IOSS. Businesses selling low value goods to customers in the EU can register in the IOSS, irrespective of the total turnover they will be making from sales to customers in the EU.

---

**F-03**　问题：我在欧盟没有公司，能自己注册 IOSS 吗？

出处：欧盟指南 4.2.10 (b) 问题 3

English answer:

You can use the simplification to collect the VAT on sales relating to low value goods and register in the IOSS in one of the EU Member States (online registration using an EU established intermediary – see sections 4.2.4 and 4.2.5 and the VAT OSS portal).

If you decide to register in the IOSS, you will only have to register in one of the EU Member States and you will be able to sell in all 27 EU Member States. You will need to appoint an intermediary who will register you in the IOSS in the Member State where he is established.

The IOSS VAT identification number issued by the Member State where you registered for IOSS (Member State of Identification) is to be used to declare all of your IOSS sales of low value goods to customers in all the EU Member States.

---

**F-04**　问题：一个 IOSS 号码能在所有欧盟国家用吗？

出处：欧盟指南 4.2.10 (b) 问题 3

English answer:

You will only have to register in one of the EU Member States and you will be able to sell in all 27 EU Member States. The IOSS VAT identification number issued by the Member State where you registered for IOSS is to be used to declare all of your IOSS sales of low value goods to customers in all the EU Member States.

When you send the goods to the customer in the EU, it is advisable to securely transmit your IOSS VAT identification number to the person who is responsible for the declaration of the goods for release for free circulation in the EU (e.g. postal operator, express carrier, customs agent) so that the VAT is not paid again to customs in the EU when the goods are imported.

---

### 第二组：150 欧元的货值怎么算

---

**F-05**　问题：判断能不能用 IOSS 的 150 欧元，是按什么价值算的？运费算不算？

出处：欧盟指南 4.2.10 (c) 问题 17

English answer:

Example 1: Invoice indicating total amount of the price paid for the goods not split between net price of the goods and transport charges. Price of the goods as indicated in the invoice: EUR 140. VAT (20%) as indicated in the invoice: EUR 28. Total invoice amount: EUR 168.

In this example, transport costs are not mentioned separately in the invoice and therefore cannot be excluded. However, the net price of the goods is not exceeding EUR 150 and therefore, IOSS can be used and no VAT or customs duties is levied at importation.

Example 2: Invoice indicating total amount of the price paid for the goods split between net price of the goods and transport charges. Price of the goods as indicated in the invoice: EUR 140. Transport charges as indicated in the invoice: EUR 20. VAT (20%) as indicated in the invoice: EUR 32. Total invoice amount: EUR 192.

In this example, transport costs are mentioned separately in the order/invoice. As such, transport costs are excluded from the intrinsic value. The intrinsic value of the goods is not exceeding EUR 150 and therefore, IOSS can be used and no VAT or customs duties is levied at importation. To be noted that VAT is applied on the total value of the sale (e.g. the EUR 160 value of the goods and the transport charges).

---

**F-06**　问题：买家用了优惠券或赶上打折，按折后价算还是原价算？

出处：欧盟指南 4.2.10 (c) 问题 20

English answer:

The intrinsic value at importation is the net price paid by the customer at the time of supply (i.e. at the time when the payment by the customer was accepted), as shown in the document accompanying the goods (i.e. commercial invoice). In case of doubt, customs authorities may request proof of payment from the customer (consignee) prior to the release of the goods for free circulation.

---

**F-07**　问题：买家付的是外币，销售时换算没超过 150 欧元，进口时因汇率波动超了怎么办？

出处：欧盟指南 4.2.10 (c) 问题 21

English answer:

Suppliers or electronic interfaces always need to make the calculation at the time of supply for the purpose of determining whether the sale of goods can be declared under the import scheme. To avoid the situation described in this question, it is recommended that the supplier or electronic interface indicates on the invoice accompanying the consignment the price in EUR, as determined at the moment of acceptance of payment. This value will be accepted by the customs authorities upon importation of the goods into the EU (unless there is suspicion of deception or fraud) and thus prevent possible double imposition of VAT upon importation.

---

**F-08**　问题：海关认为我低报了货值、实际超过 150 欧元，会怎么样？

出处：欧盟指南 4.2.10 (c) 问题 18

English answer:

For situations where despite the good faith of the supplier or the electronic interface (as deemed supplier) the intrinsic value may seem to exceed EUR 150, it is recommended that the customs authority of the Member State of importation allows the consignee to demonstrate that he has purchased the goods for a price (excluding VAT) not exceeding EUR 150 before they collect import VAT and customs duty upon clearance of the goods.

However, in case of deliberate undervaluation or any suspicion of fraud, it will not be possible for the consignee to demonstrate that he has purchased the goods for a price not exceeding EUR 150 (excluding VAT). Moreover, the IOSS scheme cannot be used. When such a situation arises, the customer (consignee) may:

- accept the delivery of goods. In this case, he will pay import VAT and, possibly, customs duty to the customs authorities, even if he already paid the VAT to the supplier or electronic interface;
- refuse the goods. In this case the usual customs practices and formalities for refusal of goods will apply.

In both cases, the customer (consignee) can contact the supplier or electronic interface to reclaim the VAT paid incorrectly at the time of supply (and possibly the amount paid for the goods in case of refusal).

---

**F-09**　问题：货值被海关重新核定，但核完之后仍然不超过 150 欧元，还能免进口增值税吗？

出处：欧盟指南 4.2.10 (c) 问题 19

English answer:

In this situation, importation of low value goods may still benefit from the exemption of VAT upon importation provided that the valid IOSS number of the taxable person claiming the import exemption is mentioned in the customs declaration. The goods will be released without payment of additional VAT to customs (the correct VAT amount has to be declared in the IOSS VAT return and has to be paid by the supplier or electronic interface or intermediary).

---

### 第三组：包裹怎么算，订单能不能拆

---

**F-10**　问题：什么情况算「一个包裹」（一次托运）？

出处：欧盟指南 4.2.10 (d) 问题 22

English answer:

Goods packed together and dispatched simultaneously by the same consignor (e.g. supplier, underlying supplier or possibly electronic interface acting as deemed supplier) to the same consignee (e.g. customer in the EU) and covered by the same transport contract (e.g. airway bill) shall be considered as a single consignment.

Consequently, goods dispatched by the same consignor to the same consignee that were ordered and shipped separately, even if arriving on the same day but as separate parcels to the postal operator or the express carrier of destination, should be considered as separate consignments, unless there is a reasonable suspicion that the consignment was split intentionally in order to avoid the payment of customs duty. In the same manner, goods ordered separately by the same person, but dispatched together, would be considered as a single consignment.

---

**F-11**　问题：一笔超过 150 欧元的订单，我分几个包裹发出，能用 IOSS 吗？

出处：欧盟指南 4.2.10 (d) 问题 25

English answer:

When purchasing several goods in a single transaction (e.g. order) this would be considered as a single supply for VAT purposes. The expectation is that the goods are dispatched/transported in a single consignment.

Since the intrinsic value of the transaction exceeds EUR 150 at the time of supply, the IOSS cannot be used by the supplier or electronic interface. Consequently, VAT should not be charged to the customer at the moment of acceptance of payment and this distance sales of imported goods should not be reported in the IOSS VAT return, even if the goods are dispatched in separate consignments.

Even if the intrinsic value of the single (partial) consignment does not exceed EUR 150, VAT will have to be calculated at the time of importation as the IOSS could not be applied at the time of supply (intrinsic value of order exceeded EUR 150) and an IOSS number must not be provided in the customs declaration. Note that the customs authorities may carry out verifications to assess whether an order or a consignment was artificially split to benefit from duty relief, case in which customs duty will be levied as well.

---

**F-12**　问题：同一个买家分几笔下单，每笔都不超过 150 欧元，但我打包成一个包裹发出去，超过 150 欧元了怎么办？

出处：欧盟指南 4.2.10 (d) 问题 24

English answer:

Each order is considered a separate supply irrespective of whether it is made by a supplier or by an underlying supplier selling via an electronic interface. As each separate order/supply does not exceed EUR 150 at the moment of acceptance of payment, the VAT should be charged by the IOSS registered supplier or respectively by the IOSS electronic interface.

When such multiple orders are packed and dispatched/transported together they will be considered as a single consignment. If suppliers or respectively electronic interfaces know that the goods of multiple orders will be dispatched or transported in a single consignment exceeding EUR 150, they should take a cautious approach and not declare the respective consignment under in the IOSS. The supplier or respectively electronic interface should reimburse the VAT collected at the moment of sale to the customer, indicating that VAT and customs duty will have to be paid upon importation into the EU. The supplier or electronic interface should keep evidence that the respective orders were dispatched in one consignment exceeding EUR 150.

---

**F-13**　问题：订单里既有普通商品，又有酒类这类要缴消费税的商品，能用 IOSS 吗？

出处：欧盟指南 4.2.10 (d) 问题 26

English answer:

The sale of the two goods in one order/transaction constitutes one single supply. Since excise goods are not covered by the IOSS, the entire order/transaction will not be subject to VAT at the moment of purchase. The fact that the value of the order/transaction does not exceed EUR 150 is not relevant. VAT will be paid upon importation irrespective whether the goods are dispatched together in the same consignment or in separate consignments.

---

### 第四组：通过平台卖货，税由谁交

---

**F-14**　问题：我通过亚马逊这类平台卖货，增值税由谁负责申报缴纳？

出处：欧盟指南 4.2.10 (b) 问题 5；3.5 场景 13

English answer:

If you only make sales of low value goods to customers in the EU via an electronic interface, you do not need to register in the IOSS. It is the electronic interface who becomes the deemed supplier for these B2C sales of goods and thus is liable to fulfil the VAT obligations regarding the sales.

When you sell goods via an electronic interface, you are deemed to supply your goods to the electronic interface and then the electronic interface makes a supply to the customer. The electronic interface (deemed supplier) is obliged to charge and collect the VAT from the customer. The electronic interface can register for the IOSS and fulfil the VAT obligations.

---

**F-15**　问题：我只通过平台卖货，自己还要不要注册 IOSS？

出处：欧盟指南 4.2.10 (b) 问题 5

English answer:

If you only make sales of low value goods to customers in the EU via an electronic interface, you do not need to register in the IOSS.

If the electronic interface registers in the IOSS and also organises the dispatch or transport of your goods to the customer, you have no specific VAT related obligations in the EU.

---

**F-16**　问题：我既通过平台卖货，也通过自己的网站卖货，该怎么处理？

出处：欧盟指南 4.2.10 (b) 问题 7

English answer:

You should keep clear evidence of the goods sold via your online shop and the goods sold via the electronic interface. If you choose to register in the IOSS for the sales made via your online shop, you should provide your own IOSS VAT identification number to person that is responsible for the declaration of the goods for release into free circulation in the EU (e.g. postal operator, express carrier, customs agent) for the goods sold via your own website.

For the goods sold via the electronic interface, you should provide to the person that is responsible for the declaration of the goods for release into free circulation in the EU the IOSS VAT identification number of the electronic interface, since you organise the transport.

If you do not register in the IOSS for the sales via your online shop, you cannot use the IOSS VAT identification number of the electronic interface for the sales made via your own website. Instead, the VAT for the goods sold via your online shop will be collected from the customer at the moment of importation in the EU.

---

**F-17**　问题：平台没有注册 IOSS，那这些货的增值税怎么办？

出处：欧盟指南 4.2.10 (b) 问题 5

English answer:

If the electronic interface does not register in IOSS, the VAT related to those goods will be collected upon importation in the EU. The VAT is due in the Member State where the dispatch or the transport of the goods ends. It shall be paid by the person designated as liable for the payment of import VAT in accordance with the national VAT legislation. Most Member States designate the customer in the EU receiving the goods to be the person liable to pay VAT. However, Member States may designate the electronic interface (deemed supplier) to be the person liable to pay VAT in these situations.

---

**F-18**　问题：平台把 IOSS 号码给我，让我自己安排发货，我要做什么？

出处：欧盟指南 4.2.10 (b) 问题 5

English answer:

If you organise the dispatch or transport to your customer, the electronic interface will give you its IOSS VAT identification number to be transmitted to the person that is responsible for the declaration of the goods for release into free circulation in the EU (e.g. postal operator, express carrier, customs agent). The latter will communicate the IOSS number to customs authorities in order to release the goods for free circulation without VAT to be paid.

You should not transmit this IOSS VAT identification number to other parties than those involved in the declaration of the goods for release for free circulation (most likely the electronic interface will impose clear commercial conditions on you before providing this IOSS VAT identification number).

---

### 第五组：注册与申报，哪些情况用得了 IOSS

---

**F-19**　问题：我是欧盟境外卖家，只通过自己的网店卖给欧盟买家，要做什么？

出处：欧盟指南 4.2.10 (b) 问题 3

English answer:

From 1 July 2021, all commercial goods imported into the EU will be subject to VAT. You can use the simplification to collect the VAT on sales relating to low value goods and register in the IOSS in one of the EU Member States (online registration using an EU established intermediary).

At the moment of sale, you will need to charge to the customer the VAT rate applicable to the goods in the Member State to which those goods will be dispatched.

Each month, your intermediary who registered you in IOSS will need to submit an IOSS VAT return by the end of the month following the reporting month (e.g. for sales in September, the IOSS VAT return is to be submitted by 31 October). The IOSS VAT return contains all the IOSS sales of low value goods in the EU broken down per Member State of destination and per VAT rate and shows the total VAT due in the EU.

If you do not register in the IOSS, the competent authorities will collect VAT upon importation of the low value goods. The customer in the EU will only receive the goods after the VAT has been paid. It could be that the representative submitting the customs declaration on behalf of the customer (e.g. postal operators or express carriers) will also charge an additional clearance fee to the customer. As the customers in the EU are used to a price that includes VAT, the payment of additional fees at the time of importation might lead to the customer refusing the package/parcel.

---

**F-20**　问题：我是欧盟境内的卖家，货从欧盟境外直邮给欧盟买家，要做什么？

出处：欧盟指南 4.2.10 (b) 问题 9、问题 10

English answer:

You can choose to register in the IOSS, however you are not obliged to have an intermediary for this purpose. In this case, the Member State in which you are established is the Member State of identification. You will charge to and collect from the customer the VAT applicable in the Member State where goods are dispatched or transported to. You will need to communicate the IOSS VAT identification number to the person responsible for the declaration of the goods for release into free circulation in the EU so that VAT is not paid again upon importation.

If you choose not to register in the IOSS, the person designated as liable to pay the import VAT in accordance with the national VAT legislation (typically the customer) will have to pay the VAT at importation in the EU and also a customs clearance fee charged by the company declaring the low value goods to customs, where applicable. These sales of goods are not to be included in the domestic VAT return.

If you sell low value goods to customers in the entire EU via your online shop, and the goods are dispatched directly from a place outside the EU, you need in each case to apply the VAT rate of the Member State to which the goods are dispatched or transported.

---

**F-21**　问题：我已经把货批量进口到欧盟的仓库了，卖出去时能用 IOSS 吗？

出处：欧盟指南 4.2.10 (b) 问题 12、问题 13

English answer:

No, you cannot register in the IOSS for these transactions.

If you import low value goods in your own name before selling them on to customers in your own country, you cannot use the IOSS for these transactions. For the importation of goods, you follow the general rules (standard or simplified procedure) applicable to the entry and import of goods into the EU. The subsequent sales to customers in the Member State in which you are established follow the normal rules for domestic supplies. You need to report those sales in your domestic VAT return.

To declare, collect, and pay the VAT on the sales made to customers in other EU Member States (i.e. where the destination of the goods is), you have two options: i) register directly in each Member State where the goods are sent to or ii) use the Union scheme (Union One Stop Shop).

---

**F-22**　问题：货放在欧盟的海关仓库里，等卖掉之后再清关，能用 IOSS 吗？

出处：欧盟指南 4.2.10 (b) 问题 14

English answer:

No, you cannot use the IOSS for these transactions. When goods are already in the EU, you cannot use the IOSS since one of the conditions to use the IOSS is that goods are dispatched or transported by or on behalf of the supplier from a third country or third territory to the customer in the EU. Moreover goods aimed at final use or consumption cannot be placed in a customs warehouse (Article 155 of the VAT Directive).

You need to remove the goods from the customs warehouse and release them for free circulation in the EU for which you will pay the VAT (i.e. at the VAT rate applicable in the Member State of importation) and customs duties (if applicable). After that, you will charge the VAT of the Member State to which the goods are dispatched/transported. If you have customers in several EU Member States, you can use the Union scheme.

---

**F-23**　问题：我从加那利群岛发货给欧盟买家，算「进口货物远程销售」还是「欧盟境内远程销售」？

出处：欧盟指南 4.2.10 (b) 问题 15

English answer:

The Canary Islands are part of the EU customs territory, but not part of the EU VAT territory (see the list of third territories in section 1.4 – Glossary). Distance sales of imported goods cover sales made from third countries, as well as third territories (see section 4.1.3).

Consequently, your sales are distance sales of imported goods for which you can register in the IOSS via an intermediary.

---

**F-24**　问题：IOSS 号码在清关时要给谁？能不能给别的服务商？

出处：欧盟指南 4.2.10 (b) 问题 3、问题 5

English answer:

When you send the goods to the customer in the EU, it is advisable to securely transmit your IOSS VAT identification number to the person who is responsible for the declaration of the goods for release for free circulation in the EU (e.g. postal operator, express carrier, customs agent) so that the VAT is not paid again to customs in the EU when the goods are imported.

You should not transmit this IOSS VAT identification number to other parties than those involved in the declaration of the goods for release for free circulation. The customs authorities will be performing their duties to ensure compliance with the customs legislation and other legislation governing movement of goods through borders without assessing or collecting any VAT.

---

### 第六组：IOSS 号码与中介

---

**F-25**　问题：怎么防止 IOSS 号码被滥用？

出处：欧盟指南 4.2.10 (g) 问题 31

English answer:

Where an electronic interface is used, it is for that electronic interface to agree on strict rules about the use of its IOSS VAT identification number by its underlying suppliers and to provide for sanctions (e.g. exclude them from the platform) against underlying suppliers not respecting these rules. The electronic interface may also negotiate transport/logistic packages for the dispatch/transport of goods sold by its underlying suppliers thus allowing it to be in contact with the transporter(s) and transmit the IOSS VAT identification number directly to it, under the same strict contractual terms.

In the medium term, the EU is working on introducing a direct exchange of information between electronic interfaces/suppliers and customs authorities. Hence, electronic interfaces would not need to rely on the diligence of the underlying suppliers.

---

**F-26**　问题：海关怎么校验 IOSS 号码？谁负责核对？

出处：欧盟指南 4.2.10 (g) 问题 33

English answer:

The validity of the IOSS VAT identification number included in a customs declaration is checked electronically by customs authorities in the IOSS VAT identification number registry/database. The database will contain all the IOSS VAT identification numbers assigned by all Member States, including their start and end validity date. The database will not be publicly available.

The control of the correct use of the IOSS VAT identification number by underlying suppliers who are in charge of the transport is in the first place the responsibility of the electronic interface to whom this number has been allocated.

The Member States will be able to control the use of the IOSS number by reconciling the amounts declared in the monthly IOSS VAT return with the monthly listing compiled from the customs declarations submitted to customs containing the total value of imports reported for each IOSS VAT identification number.

---

**F-27**　问题：报关的人是只核对号码填了没有，还是要核对号码有效？

出处：欧盟指南 4.2.10 (g) 问题 34、问题 35

English answer:

Declarants can only check the presence of the IOSS VAT identification number. They cannot check the validity as they do not have access to the IOSS VAT identification number database themselves. Only Member States through their national import system have access to that database enabling electronic verification of such numbers indicated in the import declaration.

When the IOSS VAT identification number mentioned in a customs declaration is not valid or is not provided at all, the import scheme cannot be used and the VAT exemption upon importation will not be granted. As a consequence, VAT will be levied upon importation by the customs authorities.

The responsibility for providing a valid IOSS VAT identification number lies with the supplier or the electronic interface.

---

**F-28**　问题：换中介或者换注册国之后，IOSS 号码会变吗？旧号码还能用多久？

出处：欧盟指南 4.2.10 (h) 问题 38；4.2.10 (g) 问题 36

English answer:

Yes, a change of intermediary automatically implies the attribution of a new IOSS VAT identification number for the respective taxable person. To be noted that previous IOSS VAT identification numbers have to be communicated to the Member State of identification when the taxable person requests registration via a new intermediary.

A new IOSS VAT identification number is also allocated in case the intermediary remains the same but changes Member State of identification.

The IOSS VAT identification number valid at the moment of transaction should always be used. Please note that the IOSS VAT identification number first attributed remains valid up to two months after you changed your Member State of identification. This period of maximum 2 months allows the goods rightfully sold under the IOSS being released for free circulation in the EU.

---

### 第七组：税率

---

**F-29**　问题：用 IOSS 的话，按哪个国家的税率向买家收税？

出处：欧盟指南 4.2.10 (e) 问题 28

English answer:

It is the responsibility of the supplier or electronic interface registered in the IOSS to charge the correct VAT rate applicable to the supply in the relevant Member State of consumption (e.g. the Member States to which the goods are dispatched). For the electronic interface this will be based upon the information received from the underlying supplier. The Member State of consumption will check the correctness of the rates declared in the IOSS VAT return.

---

**F-30**　问题：税率算错了谁负责？

出处：欧盟指南 4.2.10 (e) 问题 28

English answer:

It is the responsibility of the supplier or electronic interface registered in the IOSS to charge the correct VAT rate applicable to the supply in the relevant Member State of consumption. For the electronic interface this will be based upon the information received from the underlying supplier. The Member State of consumption will check the correctness of the rates declared in the IOSS VAT return.

---

### 第八组：OSS 与远程销售门槛

---

**F-31**　问题：欧盟远程销售的 1 万欧元门槛是怎么回事？

出处：欧盟指南 3.5 场景 2、场景 9

English answer:

From 1 July 2021, the threshold for distance sales of goods becomes EUR 10 000 per year and it covers all distance sales of goods to customers in all EU Member States. The previous annual EUR 35 000 distance sales threshold for each Member State (or EUR 100 000 for a limited number of Member States) disappears.

In the present scenario, since the EUR 10 000 threshold is exceeded, the place of supply of distance sale of goods is in the country to which the goods are dispatched. To report the VAT due on the distance sales of goods dispatched to Germany, Czechia and Sweden as of 1 July 2021 you have two possibilities:

a) Register in each of these Member States and declare and pay the VAT due in the national VAT return of the respective Member State; or
b) Register in the Union scheme. This is a simple online registration in the VAT One Stop Shop portal (where you are established) which is to be used for all your cross-border distance sales of goods and all supplies of services to customers in other EU Member States.

---

**F-32**　问题：没超过 1 万欧元，我还要不要做什么？

出处：欧盟指南 3.5 场景 1

English answer:

In principle, nothing changes for you in this scenario. Since you are only established in one Member State and the total value of your supplies of goods to customers in other EU Member States does not exceed EUR 10 000, they will have the same VAT treatment as your supplies to customers in Belgium.

If you want, you can choose to apply the normal rules and tax in the Member State of destination of the goods. If you choose this option, you can register for the Union scheme in the Member State where you are established. This is a simple online registration in the VAT One Stop Shop portal. If, however, you choose not to register in the OSS, you can register for VAT in the Member States of arrival of the goods.

---

**F-33**　问题：我在欧盟多个国家有仓库或有固定机构，1 万欧元门槛还适用吗？

出处：欧盟指南 3.5 场景 3

English answer:

No, the EUR 10 000 threshold is applicable only to a business established in one single Member State. Since you are established in Austria, but also have a fixed establishment in Hungary, the threshold of EUR 10 000 does not apply.

---

**F-34**　问题：1 万欧元门槛要把服务收入一起算进去吗？

出处：欧盟指南 3.5 场景 6、场景 7

English answer:

The total value of your cross-border TBE services and intra-Community distance sales of goods is EUR 9 500, thus below EUR 10 000. You can therefore apply the same VAT treatment to these cross-border supplies as for your domestic supplies. You can also opt for the place of taxation in the Member State of the customer, but this choice has to be made for both TBE supplies and distance sales of goods. You will be bound by this decision for two calendar years.

The EUR 10 000 threshold does not cover training services (or any services other than TBE services).

---

**F-35**　问题：我选了在买家所在国交税，能不能只针对货物、不含服务？

出处：欧盟指南 3.5 场景 6

English answer:

You can also opt for the place of taxation in the Member State of the customer, but this choice has to be made for both TBE supplies and distance sales of goods. You will be bound by this decision for two calendar years.

---

### 第九组：OSS 申报细节

---

**F-36**　问题：通过 OSS 申报的销售额，能不能在 OSS 申报表里抵扣进项税？

出处：欧盟指南 3.5 场景 18、场景 19

English answer:

You carry out distance sales of goods which take place in the Netherlands and France. You have to declare them in your OSS return filed in Poland (your Member State of identification) via the OSS. You cannot deduct VAT incurred in Germany via the OSS return. If you hold a stock of goods in Germany, you will normally be required to register for VAT purposes in Germany and you can deduct the input German VAT via the normal VAT return. If you are not required to register for VAT purposes in Germany, the refund of the VAT incurred in Germany may be granted based on Directive 2008/9/EC.

---

**F-37**　问题：我从 A 国发货给 B 国的买家，货从 A 国仓库出，要在哪国申报？

出处：欧盟指南 3.5 场景 15

English answer:

You have to declare these supplies of goods made from Belgium to customers in France in the Union scheme. They are intra-Community distance sales of goods (Belgium-France) whose place of supply is in France. Even though France is the Member State of identification, they are to be declared in the OSS return and may not be included in the domestic (French) VAT return.

---

### 第十组：2027 年的变化（ViDA 包）

---

**F-38**　问题：2027 年 1 月起，IOSS 和小企业免税方案能同时用吗？

出处：欧盟指南 4.2A

English answer:

A paragraph 1a has been added to Article 369m as to clarify in legislation that the IOSS scheme is incompatible with the special exemption scheme for small enterprises (SME). Before registering for the IOSS, Member States are requested to verify first a possible registration for the SME scheme. A taxable person using the special scheme for SMEs, will have to opt out of the SME scheme in case he wants to register for the IOSS scheme. Vice-versa, once a taxable person is registered for the SME exemption scheme, it will have to de-register from the IOSS scheme.

---

**F-39**　问题：2027 年 1 月起，用 OSS 申报时增值税的纳税义务发生时间有变化吗？

出处：欧盟指南 3.5（2027 部分）场景 1

English answer:

As from 1 January 2027, special national arrangements that could exist in certain Member States (Art. 66) will no longer be applicable in relation to supplies under the Union scheme (as well as the non-Union scheme).

Therefore, in all cases, the general rules will apply, regardless of your Member State of identification or the Member State where the supply is deemed to take place namely e.g. at the time when the goods are supplied (Art. 63).

For the sake of completeness, it is to be pointed out that, as regards supplies via platforms and, subsequently, the application of the deemed supplier provision, nothing has changed. The VAT will still be chargeable at the time the payment has been accepted (Art. 66a). Likewise, nothing has changed for the import scheme where the VAT remains due at the time the payment has been accepted (Art. 369n).

---

**F-40**　问题：2027 年起，欧盟远程销售的门槛有什么变化？

出处：欧盟指南 1.2（ViDA 修订说明）

English answer:

The existing threshold for intra-Community distance sales of goods will be abolished and replaced by a new EU-wide threshold of EUR 10 000 below which the supplies of goods and services are subject to VAT in the Member State of the supplier.

---

## 二、无需翻译（Amazon 卖家平台页面，本身已是中文）

这 7 条来自 `amazon-eu-vat-faq.html`。**语料本身就是中文**（亚马逊的中文本地化页），所以没有英文原文可对照。

按 ADR 的定位，「答案保留原文」这一条对它们不适用——中文页面本身就是原文。直接入库即可。

**A-01**　亚马逊要求我在 60 天内为我的卖家账户添加增值税登记号。我需要做些什么？

> 您必须提供您的公司在该国家/地区注册的有效增值税登记号，并在 60 天内将其上传至卖家平台。如果您未完成此步骤，我们可能会禁止您在该商城销售商品。

**A-02**　我是否可以上传一站式申报系统编号来代替增值税号？

> 您可以使用哪个号码取决于您的业务结构。如果您在多个国家/地区使用亚马逊物流 (FBA) 并储存库存时，您可能需要在每个国家/地区都获得增值税登记号。对于在买家所在国家/地区征税的欧盟远程销售，您可以在您居住的欧盟成员国申请或注册"一站式增值税申报"服务，规避额外的增值税登记。

**A-03**　我收到了 60 天截止期限的通知，但我的增值税登记号还在申请中，可以延期吗？

> 要申请延长截止日期，您必须告知我们您的增值税登记号正在申请中，并确认您将向相关税务机构结清以往的所有款项。在您做出这些确认后，我们将再给您 60 天的时间来提供增值税登记号。

**A-04**　如何在卖家平台中添加增值税税号？

> 请前往我们增值税登记号登记页面，然后按国家/地区输入您的可用增值税登记号。

**A-05**　什么样的增值税税号才被视为有效？

> 一般来说，注册增值税登记号的公司名称必须与卖家平台中的法定名称一致，增值税登记号方才有效。我们仅接受相关税务机构签发的增值税登记号。

**A-06**　我注册增值税号时用的公司名称和卖家平台里的不一致，该怎么办？

> 我们不接受与您的法定名称不一致的增值税登记号。您必须联系相关税务机构更新您注册增值税登记号时使用的公司名称，或者请求该税务机构重新为您颁发一个与卖家账户中的法定名称一致的增值税登记号。您也可以选择更改卖家平台企业信息中的企业名称，但这样做可能会重新触发账户验证流程。

**A-07**　在哪里可以获得有关增值税登记义务的帮助？

> 亚马逊增值税整合服务可以帮助您管理您在英国和欧盟的增值税登记和申报义务。我们会与多家税务服务提供商合作，帮助您满足英国、德国、法国、意大利、西班牙、波兰以及捷克共和国对于增值税的相关规定。

---

## 三、说明与已知局限

**问题是我按卖家视角改写的，不是语料原题。** 语料里的问句有一部分是法条解释型的（`Why was Article 14a introduced?`），卖家不会这么问。改写后的问题更贴使用场景，但**改写本身需要你审**——出题人会偏向「语料里写得很清楚的地方」，这是已知偏差。

**答案保留了原文措辞**，只做了删节（用 `……` 或直接截断到相关段落），没有改写。译文由外部大模型产出。

**`F-40` 的原文只找到一句过渡句**，ViDA 对门槛的完整修订散在指南的多个章节。这一条建议译好后人工核一遍，或者先不收录。

**本批 40 条 + Amazon 7 条 = 47 条。** 比 ADR 里说的「几十条」多，但都是语料里现成有答案的，多出来不增加维护负担。
